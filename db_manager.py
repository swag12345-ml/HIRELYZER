"""
Enhanced Database Manager for Resume Analysis System
Migrated from SQLite to Supabase PostgreSQL (psycopg2)
"""

import psycopg2
import psycopg2.extras
import pandas as pd
from datetime import datetime
import pytz
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional, List, Tuple, Dict, Any
import logging
import streamlit as st
from threading import Lock
from llm_manager import call_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Cached connection (one per Streamlit session / process) ──────────────────
@st.cache_resource
def _get_pg_connection():
    """
    Return a single cached psycopg2 connection per Streamlit worker.
    @st.cache_resource ensures the connection is NOT recreated on every rerun.
    """
    conn = psycopg2.connect(
        host=st.secrets["SUPABASE_HOST"],
        dbname=st.secrets["SUPABASE_DB"],
        user=st.secrets["SUPABASE_USER"],
        password=st.secrets["SUPABASE_PASSWORD"],
        port=st.secrets["SUPABASE_PORT"],
        connect_timeout=30,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )
    conn.autocommit = False
    logger.info("New Supabase PostgreSQL connection established (cached).")
    return conn


def _get_fresh_cursor():
    """
    Return a cursor from the cached connection.
    Reconnects automatically if the connection was dropped.
    """
    conn = _get_pg_connection()
    try:
        conn.isolation_level  # lightweight liveness check
    except Exception:
        # Connection lost – clear the cache so next call reconnects
        st.cache_resource.clear()
        conn = _get_pg_connection()
    return conn


class DatabaseManager:
    """
    Enhanced Database Manager backed by Supabase PostgreSQL.
    Public API is identical to the original SQLite version.
    """

    def __init__(self):
        self._pool_lock = Lock()
        self._initialize_database()

    # ── Internal helpers ─────────────────────────────────────────────────────

    @contextmanager
    def get_connection(self):
        """
        Context manager that yields a psycopg2 connection.
        Commits on success, rolls back on error.
        The underlying connection is the @st.cache_resource singleton.
        """
        conn = _get_fresh_cursor()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise

    def _execute(self, sql: str, params=None, fetch: str = "none"):
        """
        Run a single statement and optionally return rows.
        fetch: 'one' | 'all' | 'none'
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                if fetch == "one":
                    return cur.fetchone()
                if fetch == "all":
                    return cur.fetchall()
                return None

    def _read_df(self, sql: str, params=None) -> pd.DataFrame:
        """Execute a SELECT and return a pandas DataFrame."""
        with self.get_connection() as conn:
            try:
                df = pd.read_sql_query(sql, conn, params=params)
                return df
            except Exception as e:
                logger.error(f"read_df error: {e}")
                return pd.DataFrame()

    # ── Schema initialisation ─────────────────────────────────────────────────

    def _initialize_database(self):
        """Create tables and indexes if they don't already exist."""
        ddl = """
        CREATE TABLE IF NOT EXISTS candidates (
            id            SERIAL PRIMARY KEY,
            resume_name   TEXT NOT NULL,
            candidate_name TEXT NOT NULL,
            ats_score     INTEGER NOT NULL CHECK (ats_score BETWEEN 0 AND 100),
            edu_score     INTEGER NOT NULL CHECK (edu_score BETWEEN 0 AND 100),
            exp_score     INTEGER NOT NULL CHECK (exp_score BETWEEN 0 AND 100),
            skills_score  INTEGER NOT NULL CHECK (skills_score BETWEEN 0 AND 100),
            lang_score    INTEGER NOT NULL CHECK (lang_score BETWEEN 0 AND 100),
            keyword_score INTEGER NOT NULL CHECK (keyword_score BETWEEN 0 AND 100),
            format_score  INTEGER NOT NULL DEFAULT 0 CHECK (format_score BETWEEN 0 AND 100),
            bias_score    REAL    NOT NULL CHECK (bias_score BETWEEN 0.0 AND 1.0),
            domain        TEXT NOT NULL,
            timestamp     TIMESTAMP NOT NULL DEFAULT NOW()
        );
        ALTER TABLE candidates ADD COLUMN IF NOT EXISTS
            format_score INTEGER NOT NULL DEFAULT 0;
        CREATE INDEX IF NOT EXISTS idx_candidates_domain     ON candidates(domain);
        CREATE INDEX IF NOT EXISTS idx_candidates_ats_score  ON candidates(ats_score);
        CREATE INDEX IF NOT EXISTS idx_candidates_timestamp  ON candidates(timestamp);
        CREATE INDEX IF NOT EXISTS idx_candidates_bias_score ON candidates(bias_score);
        CREATE INDEX IF NOT EXISTS idx_candidates_domain_ats ON candidates(domain, ats_score);
        CREATE INDEX IF NOT EXISTS idx_candidates_ts_domain  ON candidates(timestamp, domain);
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(ddl)
            logger.info("Database initialised with optimized schema and indexes.")
        except Exception as e:
            logger.error(f"Schema init error: {e}")

    # ── Domain detection (unchanged logic) ───────────────────────────────────

    def detect_domain_llm(self, job_title: str, job_description: str, session=None) -> str:
        prompt = f"""
You are an expert career advisor.
Given either a job posting (title + description) OR a candidate resume (summary, skills, experience, projects),
classify the most relevant professional domain.

Job Title: {job_title}
Job / Resume Text: {job_description}

Return ONLY one domain from this list (no explanation, no extra text):
[Data Science, AI/Machine Learning, UI/UX Design, Mobile Development,
Frontend Development, Backend Development, Full Stack Development, Cybersecurity,
Cloud Engineering, DevOps/Infrastructure, Quality Assurance, Game Development,
Blockchain Development, Embedded Systems, System Architecture, Database Management,
Networking, Site Reliability Engineering, Product Management, Project Management,
Business Analysis, Technical Writing, Digital Marketing, E-commerce, Fintech,
Healthcare Tech, EdTech, IoT Development, AR/VR Development, Technical Sales,
Agile Coaching, Software Engineering]
"""
        # Provide a minimal safe session dict if none passed — prevents KeyError
        # inside call_llm() when used outside a Streamlit request context.
        if session is None:
            session = {"key_index": 0, "user_groq_key": ""}
        try:
            result = call_llm(prompt, session=session).strip()
            valid_domains = [
                "Data Science", "AI/Machine Learning", "UI/UX Design", "Mobile Development",
                "Frontend Development", "Backend Development", "Full Stack Development", "Cybersecurity",
                "Cloud Engineering", "DevOps/Infrastructure", "Quality Assurance", "Game Development",
                "Blockchain Development", "Embedded Systems", "System Architecture", "Database Management",
                "Networking", "Site Reliability Engineering", "Product Management", "Project Management",
                "Business Analysis", "Technical Writing", "Digital Marketing", "E-commerce", "Fintech",
                "Healthcare Tech", "EdTech", "IoT Development", "AR/VR Development", "Technical Sales",
                "Agile Coaching", "Software Engineering"
            ]
            return result if result in valid_domains else "Software Engineering"
        except Exception as e:
            logger.error(f"LLM domain detection failed: {e}")
            return self.detect_domain_from_title_and_description(job_title, job_description)

    def detect_domain_from_title_and_description(self, job_title: str, job_description: str) -> str:
        title = job_title.lower().strip()
        desc = job_description.lower().strip()

        replacements = {
            "cyber security": "cybersecurity", "ai engineer": "machine learning",
            "ml engineer": "machine learning", "software developer": "software engineer",
            "frontend developer": "frontend", "backend developer": "backend",
            "fullstack developer": "full stack", "devops engineer": "devops",
            "cloud engineer": "cloud", "qa engineer": "quality assurance",
            "test engineer": "quality assurance", "sre": "site reliability engineering",
            "blockchain developer": "blockchain", "game developer": "game development",
            "embedded engineer": "embedded systems", "network engineer": "networking",
            "database administrator": "database management", "dba": "database management",
            "business analyst": "business analysis", "product manager": "product management",
            "project manager": "project management", "scrum master": "agile coaching",
            "technical writer": "technical writing", "sales engineer": "technical sales",
            "solution architect": "system architecture"
        }
        for old, new in replacements.items():
            title = title.replace(old, new)
            desc = desc.replace(old, new)

        domain_scores = defaultdict(int)
        WEIGHTS = {
            "Data Science": 4, "AI/Machine Learning": 4, "UI/UX Design": 3,
            "Mobile Development": 3, "Frontend Development": 3, "Backend Development": 3,
            "Full Stack Development": 4, "Cybersecurity": 4, "Cloud Engineering": 3,
            "DevOps/Infrastructure": 3, "Quality Assurance": 3, "Game Development": 3,
            "Blockchain Development": 3, "Embedded Systems": 3, "System Architecture": 4,
            "Database Management": 3, "Networking": 3, "Site Reliability Engineering": 3,
            "Product Management": 3, "Project Management": 3, "Business Analysis": 3,
            "Technical Writing": 2, "Digital Marketing": 3, "E-commerce": 3, "Fintech": 3,
            "Healthcare Tech": 3, "EdTech": 3, "IoT Development": 3, "AR/VR Development": 3,
            "Technical Sales": 2, "Agile Coaching": 2, "Software Engineering": 2,
        }

        keywords = {
            "Data Science": [
                "data analyst","data scientist","data science","eda","pandas","numpy",
                "data analysis","statistics","data visualization","matplotlib","seaborn",
                "power bi","tableau","looker","kpi","sql","excel","dashboards","insights",
                "hypothesis testing","a/b testing","business intelligence","data wrangling",
                "feature engineering","data storytelling","exploratory analysis","data mining",
                "statistical modeling","time series","forecasting","predictive analytics",
                "analytics engineer","r programming","jupyter","databricks","spark","hadoop",
                "etl","data pipeline","data warehouse","olap","oltp","dimensional modeling",
                "data governance"
            ],
            "AI/Machine Learning": [
                "machine learning","ml engineer","deep learning","neural network","nlp",
                "computer vision","ai engineer","scikit-learn","tensorflow","pytorch","llm",
                "huggingface","xgboost","lightgbm","classification","regression",
                "reinforcement learning","transfer learning","model training","bert","gpt",
                "yolo","transformer","autoencoder","ai models","fine-tuning","zero-shot",
                "one-shot","mistral","llama","openai","langchain","vector embeddings",
                "prompt engineering","mlops","model deployment","feature store",
                "model monitoring","hyperparameter tuning","ensemble methods",
                "gradient boosting","random forest","svm","clustering","pca"
            ],
            "UI/UX Design": [
                "figma","adobe xd","sketch","wireframe","prototyping","user interface",
                "user experience","usability testing","interaction design","design system",
                "visual design","responsive design","material design","user research",
                "usability","accessibility","human-centered design","affinity diagram",
                "journey mapping","heuristic evaluation","persona","mobile-first","ux audit",
                "design tokens","design thinking","information architecture","card sorting",
                "tree testing","user testing","a/b testing design","design sprint",
                "atomic design","design ops","brand design"
            ],
            "Mobile Development": [
                "android","ios","flutter","kotlin","swift","mobile app","react native",
                "mobile application","play store","app store","firebase","mobile sdk",
                "xcode","android studio","cross-platform","native mobile","push notifications",
                "in-app purchases","mobile ui","mobile ux","apk","ipa","expo","capacitor",
                "cordova","xamarin","ionic","phonegap","mobile testing","app optimization",
                "mobile security","offline functionality","mobile analytics","app monetization",
                "mobile performance"
            ],
            "Frontend Development": [
                "frontend","html","css","javascript","react","angular","vue","typescript",
                "next.js","webpack","bootstrap","tailwind","sass","es6","responsive design",
                "web accessibility","dom","jquery","redux","vite","zustand","framer motion",
                "storybook","eslint","vitepress","pwa","single page application","csr","ssr",
                "hydration","component-based ui","web components","micro frontends","bundler",
                "transpiler","polyfill","css grid","flexbox","css animations","web performance",
                "lighthouse","core web vitals"
            ],
            "Backend Development": [
                "backend","node.js","django","flask","express","api development","sql","nosql",
                "server-side","mysql","postgresql","mongodb","rest api","graphql","java",
                "spring boot","authentication","authorization","mvc","business logic","orm",
                "database schema","asp.net","laravel","go","fastapi","nest.js","microservices",
                "websockets","rabbitmq","message broker","cron jobs","redis","elasticsearch",
                "kafka","grpc","soap","middleware","caching","load balancing","rate limiting",
                "api gateway","serverless","lambda functions"
            ],
            "Full Stack Development": [
                "full stack","fullstack","mern","mean","mevn","lamp","jamstack",
                "frontend and backend","end-to-end development","full stack developer",
                "api integration","rest api","graphql","react + node","react.js + express",
                "monolith","microservices","serverless architecture","integrated app",
                "web application","cross-functional development","component-based architecture",
                "database design","middleware","mvc","mvvm","authentication","authorization",
                "session management","cloud deployment","responsive ui","performance tuning",
                "state management","redux","context api","axios","fetch api","isomorphic",
                "universal rendering","headless cms","api-first development"
            ],
            "Cybersecurity": [
                "cybersecurity","security analyst","penetration testing","ethical hacking",
                "owasp","vulnerability","threat analysis","infosec","red team","blue team",
                "incident response","firewall","ids","ips","malware","encryption",
                "cyber threat","security operations","siem","zero-day","cyber attack",
                "kali linux","burp suite","nmap","wireshark","cve","forensics","security audit",
                "information security","compliance","ransomware","threat hunting",
                "security architecture","identity management","pki","security governance",
                "risk assessment","vulnerability management","soc"
            ],
            "Cloud Engineering": [
                "cloud","aws","azure","gcp","cloud engineer","cloud computing",
                "cloud infrastructure","cloud security","s3","ec2","cloud formation",
                "load balancer","auto scaling","cloud storage","cloud native","cloud migration",
                "eks","aks","terraform","cloudwatch","cloudtrail","iam","rds","elb","lambda",
                "azure functions","cloud functions","serverless","containers",
                "cloud architecture","multi-cloud","hybrid cloud","cloud cost optimization"
            ],
            "DevOps/Infrastructure": [
                "devops","docker","kubernetes","ci/cd","jenkins","ansible",
                "infrastructure as code","terraform","monitoring","prometheus","grafana",
                "deployment","automation","pipeline","build and release","scripting","bash",
                "shell script","site reliability","sre","argocd","helm","fluxcd","aws cli",
                "linux administration","log aggregation","observability","splunk","gitlab ci",
                "github actions","azure devops","puppet","chef","vagrant",
                "infrastructure monitoring","alerting","incident management","chaos engineering"
            ],
            "Quality Assurance": [
                "qa","quality assurance","testing","test automation","selenium","cypress",
                "test cases","test planning","bug tracking","regression testing",
                "performance testing","load testing","stress testing","api testing",
                "ui testing","unit testing","integration testing","system testing",
                "acceptance testing","test driven development","behavior driven development",
                "cucumber","jest","mocha","junit","testng","postman","jmeter","appium",
                "test management","defect management"
            ],
            "Game Development": [
                "game development","unity","unreal engine","c#","c++","game design",
                "game programming","3d modeling","animation","shader programming",
                "physics engine","game mechanics","level design","game testing","multiplayer",
                "networking","mobile games","console games","pc games","vr games","ar games",
                "game optimization","performance profiling","game analytics","monetization"
            ],
            "Blockchain Development": [
                "blockchain","cryptocurrency","smart contracts","solidity","ethereum","bitcoin",
                "defi","nft","web3","dapp","consensus algorithms","cryptography",
                "distributed ledger","mining","staking","tokenomics","metamask","truffle",
                "hardhat","ipfs","polygon","binance smart chain","hyperledger","chainlink",
                "oracles","dao","yield farming"
            ],
            "Embedded Systems": [
                "embedded systems","microcontroller","firmware","c programming","assembly",
                "real-time systems","rtos","arduino","raspberry pi","arm","pic","embedded c",
                "hardware programming","sensor integration","iot devices","low-level programming",
                "device drivers","bootloader","embedded linux","fpga","verilog","vhdl",
                "pcb design","circuit design"
            ],
            "System Architecture": [
                "system architecture","solution architect","enterprise architecture",
                "microservices","distributed systems","scalability","high availability",
                "fault tolerance","system design","architecture patterns","design patterns",
                "load balancing","caching strategies","database sharding",
                "event-driven architecture","message queues","api design","service mesh",
                "containerization","orchestration","cloud architecture"
            ],
            "Database Management": [
                "database administrator","dba","database design","sql optimization",
                "database performance","backup and recovery","replication","clustering",
                "data modeling","normalization","indexing","stored procedures","triggers",
                "database security","mysql","postgresql","oracle","sql server","mongodb",
                "cassandra","redis","elasticsearch","data warehouse","etl","olap"
            ],
            "Networking": [
                "network engineer","network administration","cisco","routing","switching",
                "tcp/ip","dns","dhcp","vpn","firewall","network security","network monitoring",
                "network troubleshooting","wan","lan","vlan","bgp","ospf","mpls","sd-wan",
                "network automation","network protocols"
            ],
            "Site Reliability Engineering": [
                "sre","site reliability","system reliability","incident management",
                "post-mortem","error budgets","sli","slo","monitoring","alerting",
                "capacity planning","performance optimization","chaos engineering",
                "disaster recovery","high availability","fault tolerance","observability"
            ],
            "Product Management": [
                "product manager","product management","product strategy","roadmap",
                "user stories","requirements gathering","stakeholder management","agile",
                "scrum","kanban","product analytics","a/b testing","user research",
                "market research","competitive analysis","go-to-market","product launch",
                "feature prioritization","backlog management","kpi","metrics"
            ],
            "Project Management": [
                "project manager","project management","pmp","agile","scrum master","kanban",
                "waterfall","risk management","resource planning","timeline","milestone",
                "deliverables","stakeholder communication","budget management",
                "team coordination","project planning","project execution","project closure",
                "change management","quality assurance","jira","confluence","ms project"
            ],
            "Business Analysis": [
                "business analyst","requirements analysis","process improvement","workflow",
                "business process","stakeholder analysis","gap analysis","use cases",
                "functional requirements","non-functional requirements","documentation",
                "process mapping","business rules","acceptance criteria",
                "user acceptance testing","change management","business intelligence",
                "data analysis","reporting"
            ],
            "Technical Writing": [
                "technical writer","documentation","api documentation","user manuals",
                "technical communication","content strategy","information architecture",
                "style guide","editing","proofreading","markdown","confluence","gitbook",
                "sphinx","doxygen","technical blogging","knowledge base"
            ],
            "Digital Marketing": [
                "digital marketing","seo","sem","social media marketing","content marketing",
                "email marketing","ppc","google ads","facebook ads","analytics",
                "conversion optimization","marketing automation","lead generation",
                "brand management","influencer marketing","affiliate marketing","growth hacking"
            ],
            "E-commerce": [
                "e-commerce","online retail","shopify","magento","woocommerce","payment gateway",
                "inventory management","order management","shipping","customer service",
                "marketplace","dropshipping","conversion rate optimization","product catalog",
                "shopping cart","checkout optimization","amazon fba"
            ],
            "Fintech": [
                "fintech","financial technology","payment processing","banking software",
                "trading systems","risk management","compliance","regulatory","kyc","aml",
                "blockchain finance","cryptocurrency","robo-advisor","insurtech",
                "lending platform","credit scoring","fraud detection","financial analytics"
            ],
            "Healthcare Tech": [
                "healthcare technology","healthtech","medical software","ehr","emr",
                "telemedicine","medical devices","hipaa","healthcare analytics","clinical trials",
                "medical imaging","bioinformatics","health informatics","patient management",
                "healthcare compliance","medical ai","digital health"
            ],
            "EdTech": [
                "edtech","educational technology","e-learning","lms","learning management",
                "online education","educational software","student information system",
                "assessment tools","educational analytics","adaptive learning","gamification",
                "virtual classroom","educational content","curriculum development"
            ],
            "IoT Development": [
                "iot","internet of things","connected devices","sensor networks","edge computing",
                "mqtt","coap","zigbee","bluetooth","wifi","embedded systems","device management",
                "iot platform","industrial iot","smart home","smart city","wearables",
                "asset tracking","predictive maintenance"
            ],
            "AR/VR Development": [
                "ar","vr","augmented reality","virtual reality","mixed reality","xr","unity 3d",
                "unreal engine","oculus","hololens","arkit","arcore","3d modeling",
                "spatial computing","immersive experience","360 video","haptic feedback",
                "motion tracking","computer vision","3d graphics"
            ],
            "Technical Sales": [
                "technical sales","sales engineer","solution selling","pre-sales",
                "technical consulting","customer success","account management",
                "product demonstration","technical presentation","proposal writing",
                "client relationship","revenue generation","sales process","crm"
            ],
            "Agile Coaching": [
                "agile coach","scrum master","agile transformation","team facilitation",
                "retrospectives","sprint planning","daily standups","agile ceremonies",
                "continuous improvement","change management","team dynamics","agile metrics",
                "coaching","mentoring","organizational change"
            ],
            "Software Engineering": [
                "software engineer","web developer","developer","programmer","object oriented",
                "design patterns","agile","scrum","git","version control","unit testing",
                "integration testing","debugging","code review","system design","tdd","bdd",
                "pair programming","refactoring","uml","dev environment","ide","algorithms",
                "data structures","software architecture","clean code"
            ],
        }

        for domain, kws in keywords.items():
            title_hits = sum(1 for kw in kws if kw in title)
            desc_hits = sum(1 for kw in kws if kw in desc)
            domain_scores[domain] = (4 * title_hits + 1 * desc_hits) * WEIGHTS[domain]

        frontend_hits = sum(1 for kw in keywords["Frontend Development"] if kw in title or kw in desc)
        backend_hits = sum(1 for kw in keywords["Backend Development"] if kw in title or kw in desc)
        fullstack_mentioned = any(t in title or t in desc for t in ["full stack", "fullstack", "full-stack"])
        if fullstack_mentioned:
            domain_scores["Full Stack Development"] += 15
        if frontend_hits >= 4 and backend_hits >= 4:
            domain_scores["Full Stack Development"] += 12

        domain_boosts = {
            "AI/Machine Learning": ["ai", "ml", "machine learning", "artificial intelligence"],
            "Cybersecurity": ["security", "cyber", "infosec"],
            "Cloud Engineering": ["cloud", "aws", "azure", "gcp"],
            "Mobile Development": ["mobile", "android", "ios", "app"],
            "Game Development": ["game", "unity", "unreal"],
            "Blockchain Development": ["blockchain", "crypto", "web3", "defi"],
            "IoT Development": ["iot", "embedded", "sensor"],
            "AR/VR Development": ["ar", "vr", "augmented", "virtual reality"],
        }
        for domain, boost_terms in domain_boosts.items():
            if any(t in title for t in boost_terms):
                domain_scores[domain] += 8
            if any(t in desc for t in boost_terms):
                domain_scores[domain] += 3

        if len(desc.split()) < 8:
            strong_keywords = ["full stack developer", "mobile developer", "android developer", "ios developer"]
            if not any(k in title or k in desc for k in strong_keywords):
                for domain in domain_scores:
                    desc_hits = sum(1 for kw in keywords[domain] if kw in desc)
                    domain_scores[domain] = max(0, domain_scores[domain] - (desc_hits * WEIGHTS[domain] * 0.5))

        if domain_scores:
            top_domain = max(domain_scores, key=domain_scores.get)
            top_score = domain_scores[top_domain]
            if top_score >= 8:
                if "full stack developer" in title:
                    return "Full Stack Development"
                if "mobile developer" in title or "android developer" in title or "ios developer" in title:
                    return "Mobile Development"
                return top_domain
        return "Software Engineering"

    def get_domain_similarity(self, resume_domain: str, job_domain: str) -> float:
        resume_domain = resume_domain.strip().lower()
        job_domain = job_domain.strip().lower()

        normalization = {
            "frontend": "frontend development", "backend": "backend development",
            "fullstack": "full stack development", "full-stack": "full stack development",
            "ui/ux": "ui/ux design", "ux/ui": "ui/ux design",
            "software developer": "software engineering",
            "mobile developer": "mobile development",
            "android developer": "mobile development",
            "ios developer": "mobile development",
            "ai": "ai/machine learning", "machine learning": "ai/machine learning",
            "ml": "ai/machine learning", "artificial intelligence": "ai/machine learning",
            "cloud": "cloud engineering", "cloud engineer": "cloud engineering",
            "devops": "devops/infrastructure", "devops engineer": "devops/infrastructure",
            "cyber security": "cybersecurity", "cybersecurity engineer": "cybersecurity",
            "security analyst": "cybersecurity", "qa": "quality assurance",
            "test engineer": "quality assurance", "sre": "site reliability engineering",
            "dba": "database management", "database administrator": "database management",
            "product manager": "product management", "project manager": "project management",
            "business analyst": "business analysis", "technical writer": "technical writing",
            "game developer": "game development", "blockchain developer": "blockchain development",
        }
        resume_domain = normalization.get(resume_domain, resume_domain)
        job_domain = normalization.get(job_domain, job_domain)

        similarity_map = {
            ("full stack development", "frontend development"): 0.85,
            ("full stack development", "backend development"): 0.85,
            ("full stack development", "ui/ux design"): 0.70,
            ("full stack development", "mobile development"): 0.65,
            ("full stack development", "software engineering"): 0.80,
            ("frontend development", "ui/ux design"): 0.90,
            ("frontend development", "mobile development"): 0.70,
            ("frontend development", "software engineering"): 0.75,
            ("frontend development", "backend development"): 0.60,
            ("backend development", "database management"): 0.80,
            ("backend development", "cloud engineering"): 0.75,
            ("backend development", "devops/infrastructure"): 0.70,
            ("backend development", "system architecture"): 0.85,
            ("backend development", "software engineering"): 0.80,
            ("data science", "ai/machine learning"): 0.95,
            ("data science", "business analysis"): 0.70,
            ("ai/machine learning", "data science"): 0.95,
            ("ai/machine learning", "software engineering"): 0.65,
            ("cloud engineering", "devops/infrastructure"): 0.90,
            ("cloud engineering", "system architecture"): 0.80,
            ("cloud engineering", "site reliability engineering"): 0.85,
            ("devops/infrastructure", "site reliability engineering"): 0.90,
            ("devops/infrastructure", "system architecture"): 0.75,
            ("cybersecurity", "devops/infrastructure"): 0.70,
            ("cybersecurity", "cloud engineering"): 0.75,
            ("cybersecurity", "networking"): 0.80,
            ("cybersecurity", "system architecture"): 0.65,
            ("mobile development", "ui/ux design"): 0.75,
            ("mobile development", "software engineering"): 0.70,
            ("mobile development", "game development"): 0.60,
            ("quality assurance", "software engineering"): 0.75,
            ("quality assurance", "devops/infrastructure"): 0.65,
            ("quality assurance", "system architecture"): 0.60,
            ("product management", "business analysis"): 0.80,
            ("product management", "project management"): 0.75,
            ("project management", "agile coaching"): 0.85,
            ("business analysis", "data science"): 0.65,
            ("game development", "software engineering"): 0.70,
            ("blockchain development", "software engineering"): 0.70,
            ("blockchain development", "cybersecurity"): 0.65,
            ("embedded systems", "iot development"): 0.90,
            ("ar/vr development", "game development"): 0.80,
            ("ar/vr development", "mobile development"): 0.70,
            ("database management", "data science"): 0.75,
            ("database management", "system architecture"): 0.70,
            ("database management", "backend development"): 0.80,
            ("system architecture", "software engineering"): 0.85,
            ("system architecture", "cloud engineering"): 0.80,
            ("system architecture", "backend development"): 0.85,
            ("networking", "cybersecurity"): 0.80,
            ("networking", "devops/infrastructure"): 0.75,
            ("networking", "system architecture"): 0.70,
            ("fintech", "software engineering"): 0.70,
            ("fintech", "backend development"): 0.75,
            ("fintech", "cybersecurity"): 0.70,
            ("healthcare tech", "software engineering"): 0.70,
            ("edtech", "software engineering"): 0.70,
            ("e-commerce", "full stack development"): 0.80,
            ("e-commerce", "backend development"): 0.75,
            ("technical sales", "product management"): 0.65,
            ("technical writing", "business analysis"): 0.60,
            ("digital marketing", "business analysis"): 0.55,
            ("software engineering", "full stack development"): 0.80,
            ("software engineering", "frontend development"): 0.75,
            ("software engineering", "backend development"): 0.80,
            ("software engineering", "mobile development"): 0.70,
            ("software engineering", "game development"): 0.70,
            ("software engineering", "quality assurance"): 0.75,
        }

        if resume_domain == job_domain:
            return 1.0
        similarity = (similarity_map.get((resume_domain, job_domain)) or
                      similarity_map.get((job_domain, resume_domain)))
        if similarity:
            return similarity

        tech_domains = {"software engineering","full stack development","frontend development",
                        "backend development","mobile development","game development",
                        "blockchain development","embedded systems","iot development"}
        data_domains = {"data science","ai/machine learning","business analysis"}
        infrastructure_domains = {"cloud engineering","devops/infrastructure","site reliability engineering",
                                  "system architecture","database management","networking","cybersecurity"}
        management_domains = {"product management","project management","business analysis","agile coaching"}
        design_domains = {"ui/ux design","ar/vr development"}

        categories = [tech_domains, data_domains, infrastructure_domains, management_domains, design_domains]
        for category in categories:
            if resume_domain in category and job_domain in category:
                return 0.50
        if ((resume_domain in tech_domains and job_domain in infrastructure_domains) or
                (resume_domain in infrastructure_domains and job_domain in tech_domains)):
            return 0.45
        if ((resume_domain in data_domains and job_domain in tech_domains) or
                (resume_domain in tech_domains and job_domain in data_domains)):
            return 0.40
        return 0.25

    # ── CRUD operations ───────────────────────────────────────────────────────

    def insert_candidate(self, data: Tuple, job_title: str = "", job_description: str = "") -> int:
        try:
            local_tz = pytz.timezone("Asia/Kolkata")
            local_time = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")
            detected_domain = self.detect_domain_from_title_and_description(job_title, job_description)

            if len(data) < 9:
                raise ValueError(f"Expected at least 9 data fields, got {len(data)}")

            # Unpack core fields (positions 0–8 are unchanged for backward compat)
            # Position 9 is the new optional format_score (defaults to 0 if not supplied)
            resume_name    = data[0]
            candidate_name = data[1]
            ats_score      = data[2]
            edu_score      = data[3]
            exp_score      = data[4]
            skills_score   = data[5]
            lang_score     = data[6]
            keyword_score  = data[7]
            bias_score     = data[8]
            format_score   = int(data[9]) if len(data) >= 10 else 0

            # Validate all integer scores (0–100)
            for name, val in [
                ("ats_score", ats_score), ("edu_score", edu_score),
                ("exp_score", exp_score), ("skills_score", skills_score),
                ("lang_score", lang_score), ("keyword_score", keyword_score),
                ("format_score", format_score),
            ]:
                if not isinstance(val, (int, float)) or not (0 <= val <= 100):
                    raise ValueError(f"{name} must be between 0 and 100, got {val}")

            if not isinstance(bias_score, (int, float)) or not (0.0 <= bias_score <= 1.0):
                raise ValueError(f"Bias score must be between 0.0 and 1.0, got {bias_score}")

            sql = """
                INSERT INTO candidates (
                    resume_name, candidate_name, ats_score, edu_score, exp_score,
                    skills_score, lang_score, keyword_score, format_score, bias_score,
                    domain, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            params = (
                resume_name, candidate_name, ats_score, edu_score, exp_score,
                skills_score, lang_score, keyword_score, format_score, bias_score,
                detected_domain, local_time,
            )
            row = self._execute(sql, params, fetch="one")
            candidate_id = row["id"] if row else None
            logger.info(f"Inserted candidate with ID: {candidate_id}")
            return candidate_id
        except Exception as e:
            logger.error(f"Error inserting candidate: {e}")
            raise

    def get_top_domains_by_score(self, limit: int = 5) -> List[Tuple]:
        try:
            sql = """
                SELECT domain, ROUND(AVG(ats_score)::numeric, 2) AS avg_score, COUNT(*) AS count
                FROM candidates
                GROUP BY domain
                HAVING COUNT(*) >= 1
                ORDER BY avg_score DESC
                LIMIT %s
            """
            rows = self._execute(sql, (limit,), fetch="all")
            return [(r["domain"], float(r["avg_score"]), r["count"]) for r in (rows or [])]
        except Exception as e:
            logger.error(f"Error getting top domains: {e}")
            return []

    def get_resume_count_by_day(self) -> pd.DataFrame:
        try:
            sql = """
                SELECT DATE(timestamp) AS day, COUNT(*) AS count
                FROM candidates
                GROUP BY DATE(timestamp)
                ORDER BY DATE(timestamp) DESC
                LIMIT 365
            """
            return self._read_df(sql)
        except Exception as e:
            logger.error(f"Error getting resume count by day: {e}")
            return pd.DataFrame()

    def get_average_ats_by_domain(self) -> pd.DataFrame:
        try:
            sql = """
                SELECT domain,
                       ROUND(AVG(ats_score)::numeric, 2) AS avg_ats_score,
                       COUNT(*) AS candidate_count
                FROM candidates
                GROUP BY domain
                HAVING COUNT(*) >= 1
                ORDER BY avg_ats_score DESC
            """
            return self._read_df(sql)
        except Exception as e:
            logger.error(f"Error getting average ATS by domain: {e}")
            return pd.DataFrame()

    def get_domain_distribution(self) -> pd.DataFrame:
        try:
            sql = """
                SELECT domain,
                       COUNT(*) AS count,
                       ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM candidates), 2) AS percentage
                FROM candidates
                GROUP BY domain
                ORDER BY count DESC
            """
            return self._read_df(sql)
        except Exception as e:
            logger.error(f"Error getting domain distribution: {e}")
            return pd.DataFrame()

    def filter_candidates_by_date(self, start: str, end: str) -> pd.DataFrame:
        try:
            datetime.strptime(start, '%Y-%m-%d')
            datetime.strptime(end, '%Y-%m-%d')
            sql = """
                SELECT * FROM candidates
                WHERE DATE(timestamp) BETWEEN %s AND %s
                ORDER BY timestamp DESC
            """
            return self._read_df(sql, params=(start, end))
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error filtering candidates by date: {e}")
            return pd.DataFrame()

    def delete_candidate_by_id(self, candidate_id: int) -> bool:
        try:
            sql = "DELETE FROM candidates WHERE id = %s"
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (candidate_id,))
                    deleted = cur.rowcount
            if deleted > 0:
                logger.info(f"Deleted candidate with ID: {candidate_id}")
                return True
            logger.warning(f"No candidate found with ID: {candidate_id}")
            return False
        except Exception as e:
            logger.error(f"Error deleting candidate: {e}")
            return False

    def get_all_candidates(self, bias_threshold: Optional[float] = None,
                           min_ats: Optional[int] = None,
                           limit: Optional[int] = None,
                           offset: int = 0) -> pd.DataFrame:
        try:
            sql = "SELECT * FROM candidates WHERE 1=1"
            params: list = []
            if bias_threshold is not None:
                sql += " AND bias_score >= %s"
                params.append(bias_threshold)
            if min_ats is not None:
                sql += " AND ats_score >= %s"
                params.append(min_ats)
            sql += " ORDER BY timestamp DESC"
            if limit is not None:
                sql += " LIMIT %s OFFSET %s"
                params.extend([limit, offset])
            return self._read_df(sql, params=params if params else None)
        except Exception as e:
            logger.error(f"Error getting all candidates: {e}")
            return pd.DataFrame()

    def export_to_csv(self, filepath: str = "candidates_export.csv",
                      filters: Optional[Dict[str, Any]] = None) -> bool:
        try:
            sql = "SELECT * FROM candidates WHERE 1=1"
            params: list = []
            if filters:
                if 'min_ats' in filters:
                    sql += " AND ats_score >= %s"
                    params.append(filters['min_ats'])
                if 'domain' in filters:
                    sql += " AND domain = %s"
                    params.append(filters['domain'])
                if 'start_date' in filters:
                    sql += " AND DATE(timestamp) >= %s"
                    params.append(filters['start_date'])
                if 'end_date' in filters:
                    sql += " AND DATE(timestamp) <= %s"
                    params.append(filters['end_date'])
            sql += " ORDER BY timestamp DESC"
            df = self._read_df(sql, params=params if params else None)
            df.to_csv(filepath, index=False)
            logger.info(f"Exported {len(df)} records to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

    def get_candidate_by_id(self, candidate_id: int) -> pd.DataFrame:
        try:
            sql = "SELECT * FROM candidates WHERE id = %s"
            return self._read_df(sql, params=(candidate_id,))
        except Exception as e:
            logger.error(f"Error getting candidate by ID: {e}")
            return pd.DataFrame()

    def get_bias_distribution(self, threshold: float = 0.6) -> pd.DataFrame:
        try:
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("Threshold must be between 0.0 and 1.0")
            sql = """
                SELECT
                    CASE WHEN bias_score >= %s THEN 'Biased' ELSE 'Fair' END AS bias_category,
                    COUNT(*) AS count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM candidates), 2) AS percentage
                FROM candidates
                GROUP BY bias_category
            """
            return self._read_df(sql, params=(threshold,))
        except Exception as e:
            logger.error(f"Error getting bias distribution: {e}")
            return pd.DataFrame()

    def get_daily_ats_stats(self, days_limit: int = 90) -> pd.DataFrame:
        try:
            # Use parameterised interval to avoid f-string SQL injection risk
            sql = """
                SELECT DATE(timestamp) AS date,
                       ROUND(AVG(ats_score)::numeric, 2) AS avg_ats,
                       COUNT(*) AS daily_count
                FROM candidates
                WHERE DATE(timestamp) >= CURRENT_DATE - (%(days)s || ' days')::INTERVAL
                GROUP BY DATE(timestamp)
                ORDER BY DATE(timestamp)
            """
            return self._read_df(sql, params={"days": days_limit})
        except Exception as e:
            logger.error(f"Error getting daily ATS stats: {e}")
            return pd.DataFrame()

    def get_flagged_candidates(self, threshold: float = 0.6) -> pd.DataFrame:
        try:
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("Threshold must be between 0.0 and 1.0")
            sql = """
                SELECT resume_name, candidate_name, ats_score, bias_score, domain, timestamp
                FROM candidates
                WHERE bias_score > %s
                ORDER BY bias_score DESC
            """
            return self._read_df(sql, params=(threshold,))
        except Exception as e:
            logger.error(f"Error getting flagged candidates: {e}")
            return pd.DataFrame()

    def get_domain_performance_stats(self) -> pd.DataFrame:
        try:
            sql = """
                SELECT
                    domain,
                    COUNT(*) AS total_candidates,
                    ROUND(AVG(ats_score)::numeric, 2)     AS avg_ats_score,
                    ROUND(AVG(edu_score)::numeric, 2)     AS avg_edu_score,
                    ROUND(AVG(exp_score)::numeric, 2)     AS avg_exp_score,
                    ROUND(AVG(skills_score)::numeric, 2)  AS avg_skills_score,
                    ROUND(AVG(lang_score)::numeric, 2)    AS avg_lang_score,
                    ROUND(AVG(keyword_score)::numeric, 2) AS avg_keyword_score,
                    ROUND(AVG(format_score)::numeric, 2)  AS avg_format_score,
                    ROUND(AVG(bias_score)::numeric, 3)    AS avg_bias_score,
                    MAX(ats_score) AS max_ats_score,
                    MIN(ats_score) AS min_ats_score,
                    ROUND((MAX(ats_score) - MIN(ats_score))::numeric, 2) AS score_range
                FROM candidates
                GROUP BY domain
                HAVING COUNT(*) >= 1
                ORDER BY avg_ats_score DESC
            """
            return self._read_df(sql)
        except Exception as e:
            logger.error(f"Error getting domain performance stats: {e}")
            return pd.DataFrame()

    def analyze_domain_transitions(self) -> pd.DataFrame:
        try:
            sql = """
                SELECT
                    domain,
                    COUNT(*) AS frequency,
                    ROUND(AVG(ats_score)::numeric, 2)  AS avg_performance,
                    ROUND(AVG(bias_score)::numeric, 3) AS avg_bias,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM candidates), 2) AS percentage
                FROM candidates
                GROUP BY domain
                HAVING COUNT(*) >= 1
                ORDER BY frequency DESC
            """
            return self._read_df(sql)
        except Exception as e:
            logger.error(f"Error analyzing domain transitions: {e}")
            return pd.DataFrame()

    def get_database_stats(self) -> Dict[str, Any]:
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT COUNT(*) AS cnt FROM candidates")
                    total_candidates = cur.fetchone()["cnt"]

                    cur.execute("""
                        SELECT
                            ROUND(AVG(ats_score)::numeric, 2)  AS avg_ats,
                            ROUND(AVG(bias_score)::numeric, 3) AS avg_bias,
                            COUNT(DISTINCT domain)             AS unique_domains
                        FROM candidates
                    """)
                    avg_stats = cur.fetchone()

                    cur.execute("""
                        SELECT
                            MIN(DATE(timestamp)) AS earliest_date,
                            MAX(DATE(timestamp)) AS latest_date
                        FROM candidates
                    """)
                    date_range = cur.fetchone()

            return {
                'total_candidates': total_candidates,
                'avg_ats_score': float(avg_stats["avg_ats"]) if avg_stats["avg_ats"] else 0,
                'avg_bias_score': float(avg_stats["avg_bias"]) if avg_stats["avg_bias"] else 0,
                'unique_domains': avg_stats["unique_domains"] if avg_stats["unique_domains"] else 0,
                'earliest_date': str(date_range["earliest_date"]) if date_range["earliest_date"] else None,
                'latest_date': str(date_range["latest_date"]) if date_range["latest_date"] else None,
                'database_size_mb': 0,   # not applicable for hosted Supabase
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

    def cleanup_old_records(self, days_to_keep: int = 365) -> int:
        try:
            sql = "DELETE FROM candidates WHERE DATE(timestamp) < CURRENT_DATE - (%(days)s || ' days')::INTERVAL"
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, {"days": days_to_keep})
                    deleted = cur.rowcount
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old records")
            return deleted
        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")
            return 0

    def close_all_connections(self):
        """No-op: connection lifecycle is managed by @st.cache_resource."""
        logger.info("close_all_connections called — connection managed by cache_resource.")


# ── Global instance (backward compatibility) ─────────────────────────────────
db_manager = DatabaseManager()


# ── Module-level wrappers (backward compatibility) ────────────────────────────
def detect_domain_from_title_and_description(job_title: str, job_description: str) -> str:
    return db_manager.detect_domain_from_title_and_description(job_title, job_description)

def get_domain_similarity(resume_domain: str, job_domain: str) -> float:
    return db_manager.get_domain_similarity(resume_domain, job_domain)

def insert_candidate(data: tuple, job_title: str = "", job_description: str = ""):
    return db_manager.insert_candidate(data, job_title, job_description)

def get_top_domains_by_score(limit: int = 5) -> list:
    return db_manager.get_top_domains_by_score(limit)

def get_resume_count_by_day():
    return db_manager.get_resume_count_by_day()

def get_average_ats_by_domain():
    return db_manager.get_average_ats_by_domain()

def get_domain_distribution():
    return db_manager.get_domain_distribution()

def filter_candidates_by_date(start: str, end: str):
    return db_manager.filter_candidates_by_date(start, end)

def delete_candidate_by_id(candidate_id: int):
    return db_manager.delete_candidate_by_id(candidate_id)

def get_all_candidates(bias_threshold: float = None, min_ats: int = None):
    return db_manager.get_all_candidates(bias_threshold, min_ats)

def export_to_csv(filepath: str = "candidates_export.csv"):
    return db_manager.export_to_csv(filepath)

def get_candidate_by_id(candidate_id: int):
    return db_manager.get_candidate_by_id(candidate_id)

def get_bias_distribution(threshold: float = 0.6):
    return db_manager.get_bias_distribution(threshold)

def get_daily_ats_stats(days_limit: int = 90):
    return db_manager.get_daily_ats_stats(days_limit)

def get_flagged_candidates(threshold: float = 0.6):
    return db_manager.get_flagged_candidates(threshold)

def get_domain_performance_stats():
    return db_manager.get_domain_performance_stats()

def analyze_domain_transitions():
    return db_manager.analyze_domain_transitions()

def get_database_stats():
    return db_manager.get_database_stats()

def cleanup_old_records(days_to_keep: int = 365):
    return db_manager.cleanup_old_records(days_to_keep)

def close_all_connections():
    return db_manager.close_all_connections()


if __name__ == "__main__":
    print("Database Manager (Supabase PostgreSQL) initialised successfully!")
    stats = get_database_stats()
    print(f"Database Statistics: {stats}")
