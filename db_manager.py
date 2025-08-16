import sqlite3
import pandas as pd
from datetime import datetime
import pytz
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional, List, Tuple, Dict, Any
import logging
from threading import Lock
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Enhanced Database Manager with connection pooling and optimized queries
    for handling large-scale user structures
    """
    
    def __init__(self, db_path: str = "resume_data.db", pool_size: int = 10):
        self.db_path = db_path
        self.pool_size = pool_size
        self._connection_pool = []
        self._pool_lock = Lock()
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize database with optimized schema and indexes"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create main candidates table with optimized schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_name TEXT NOT NULL,
                    candidate_name TEXT NOT NULL,
                    ats_score INTEGER NOT NULL CHECK(ats_score >= 0 AND ats_score <= 100),
                    edu_score INTEGER NOT NULL CHECK(edu_score >= 0 AND edu_score <= 100),
                    exp_score INTEGER NOT NULL CHECK(exp_score >= 0 AND exp_score <= 100),
                    skills_score INTEGER NOT NULL CHECK(skills_score >= 0 AND skills_score <= 100),
                    lang_score INTEGER NOT NULL CHECK(lang_score >= 0 AND lang_score <= 100),
                    keyword_score INTEGER NOT NULL CHECK(keyword_score >= 0 AND keyword_score <= 100),
                    bias_score REAL NOT NULL CHECK(bias_score >= 0.0 AND bias_score <= 1.0),
                    domain TEXT NOT NULL,
                    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create optimized indexes for better query performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_candidates_domain ON candidates(domain)",
                "CREATE INDEX IF NOT EXISTS idx_candidates_ats_score ON candidates(ats_score)",
                "CREATE INDEX IF NOT EXISTS idx_candidates_timestamp ON candidates(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_candidates_bias_score ON candidates(bias_score)",
                "CREATE INDEX IF NOT EXISTS idx_candidates_domain_ats ON candidates(domain, ats_score)",
                "CREATE INDEX IF NOT EXISTS idx_candidates_timestamp_domain ON candidates(timestamp, domain)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            logger.info("Database initialized with optimized schema and indexes")

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections with connection pooling
        """
        conn = None
        try:
            with self._pool_lock:
                if self._connection_pool:
                    conn = self._connection_pool.pop()
                else:
                    conn = sqlite3.connect(
                        self.db_path, 
                        check_same_thread=False,
                        timeout=30.0  # 30 second timeout for large operations
                    )
                    # Optimize SQLite settings for performance
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA cache_size=10000")
                    conn.execute("PRAGMA temp_store=MEMORY")
            
            yield conn
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                with self._pool_lock:
                    if len(self._connection_pool) < self.pool_size:
                        self._connection_pool.append(conn)
                    else:
                        conn.close()

    # ------------------ New Resume Methods ------------------

    def insert_resume_data(self, resume_name: str, candidate_name: str, domain: str,
                           ats_score: int, edu_score: int, exp_score: int,
                           skills_score: int, lang_score: int, keyword_score: int,
                           bias_score: float):
        """
        Insert a new resume analysis record with full ATS breakdown.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO candidates 
                (resume_name, candidate_name, domain, ats_score, edu_score, exp_score,
                 skills_score, lang_score, keyword_score, bias_score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                resume_name, candidate_name, domain,
                ats_score, edu_score, exp_score,
                skills_score, lang_score, keyword_score,
                bias_score
            ))
            conn.commit()

    def get_existing_analysis(self, candidate_name: str, resume_name: str):
        """
        Fetch the most recent analysis for a given candidate + resume name.
        Returns (ats_score, edu_score, exp_score, skills_score, lang_score, keyword_score, bias_score, timestamp)
        if found, else None.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ats_score, edu_score, exp_score, skills_score, lang_score, keyword_score, bias_score, timestamp
                FROM candidates
                WHERE candidate_name = ? AND resume_name = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (candidate_name, resume_name))
            return cursor.fetchone()

    def get_user_resume_history(self, candidate_name: str, limit: int = 10):
        """
        Fetch the last N resumes analyzed for a user.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT domain, ats_score, edu_score, exp_score, skills_score, lang_score, keyword_score, bias_score, timestamp
                FROM candidates
                WHERE candidate_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (candidate_name, limit))
            return cursor.fetchall()


                        

    def detect_domain_from_title_and_description(self, job_title: str, job_description: str) -> str:
        """
        Enhanced Domain Detection with 25+ Professional Domains
        Optimized for better performance with cached keyword lookups
        """
        title = job_title.lower().strip()
        desc = job_description.lower().strip()

        # Enhanced normalization with more synonyms
        replacements = {
            "cyber security": "cybersecurity",
            "ai engineer": "machine learning",
            "ml engineer": "machine learning",
            "software developer": "software engineer",
            "frontend developer": "frontend",
            "backend developer": "backend",
            "fullstack developer": "full stack",
            "devops engineer": "devops",
            "cloud engineer": "cloud",
            "qa engineer": "quality assurance",
            "test engineer": "quality assurance",
            "sre": "site reliability engineering",
            "blockchain developer": "blockchain",
            "game developer": "game development",
            "embedded engineer": "embedded systems",
            "network engineer": "networking",
            "database administrator": "database management",
            "dba": "database management",
            "business analyst": "business analysis",
            "product manager": "product management",
            "project manager": "project management",
            "scrum master": "agile coaching",
            "technical writer": "technical writing",
            "sales engineer": "technical sales",
            "solution architect": "system architecture"
        }
        
        for old, new in replacements.items():
            title = title.replace(old, new)
            desc = desc.replace(old, new)

        domain_scores = defaultdict(int)

        # Enhanced weights for better domain differentiation
        WEIGHTS = {
            "Data Science": 4,
            "AI/Machine Learning": 4,
            "UI/UX Design": 3,
            "Mobile Development": 3,
            "Frontend Development": 3,
            "Backend Development": 3,
            "Full Stack Development": 4,
            "Cybersecurity": 4,
            "Cloud Engineering": 3,
            "DevOps/Infrastructure": 3,
            "Quality Assurance": 3,
            "Game Development": 3,
            "Blockchain Development": 3,
            "Embedded Systems": 3,
            "System Architecture": 4,
            "Database Management": 3,
            "Networking": 3,
            "Site Reliability Engineering": 3,
            "Product Management": 3,
            "Project Management": 3,
            "Business Analysis": 3,
            "Technical Writing": 2,
            "Digital Marketing": 3,
            "E-commerce": 3,
            "Fintech": 3,
            "Healthcare Tech": 3,
            "EdTech": 3,
            "IoT Development": 3,
            "AR/VR Development": 3,
            "Technical Sales": 2,
            "Agile Coaching": 2,
            "Software Engineering": 2,  # General fallback
        }

        # Comprehensive keyword mapping for 30+ domains
        keywords = {
            "Data Science": [
                "data analyst", "data scientist", "data science", "eda", "pandas", "numpy",
                "data analysis", "statistics", "data visualization", "matplotlib", "seaborn",
                "power bi", "tableau", "looker", "kpi", "sql", "excel", "dashboards",
                "insights", "hypothesis testing", "a/b testing", "business intelligence", "data wrangling",
                "feature engineering", "data storytelling", "exploratory analysis", "data mining",
                "statistical modeling", "time series", "forecasting", "predictive analytics", "analytics engineer",
                "r programming", "jupyter", "databricks", "spark", "hadoop", "etl", "data pipeline",
                "data warehouse", "olap", "oltp", "dimensional modeling", "data governance"
            ],
            
            "AI/Machine Learning": [
                "machine learning", "ml engineer", "deep learning", "neural network",
                "nlp", "computer vision", "ai engineer", "scikit-learn", "tensorflow", "pytorch",
                "llm", "huggingface", "xgboost", "lightgbm", "classification", "regression",
                "reinforcement learning", "transfer learning", "model training", "bert", "gpt",
                "yolo", "transformer", "autoencoder", "ai models", "fine-tuning", "zero-shot", "one-shot",
                "mistral", "llama", "openai", "langchain", "vector embeddings", "prompt engineering",
                "mlops", "model deployment", "feature store", "model monitoring", "hyperparameter tuning",
                "ensemble methods", "gradient boosting", "random forest", "svm", "clustering", "pca"
            ],
            
            "UI/UX Design": [
                "ui", "ux", "figma", "designer", "user interface", "user experience",
                "adobe xd", "sketch", "wireframe", "prototyping", "interaction design",
                "user research", "usability", "design system", "visual design", "accessibility",
                "human-centered design", "affinity diagram", "journey mapping", "heuristic evaluation",
                "persona", "responsive design", "mobile-first", "ux audit", "design tokens", "design thinking",
                "information architecture", "card sorting", "tree testing", "user testing", "a/b testing design",
                "design sprint", "atomic design", "material design", "design ops", "brand design"
            ],
            
            "Mobile Development": [
                "android", "ios", "flutter", "kotlin", "swift", "mobile app", "react native",
                "mobile application", "play store", "app store", "firebase", "mobile sdk",
                "xcode", "android studio", "cross-platform", "native mobile", "push notifications",
                "in-app purchases", "mobile ui", "mobile ux", "apk", "ipa", "expo", "capacitor", "cordova",
                "xamarin", "ionic", "phonegap", "mobile testing", "app optimization", "mobile security",
                "offline functionality", "mobile analytics", "app monetization", "mobile performance"
            ],
            
            "Frontend Development": [
                "frontend", "html", "css", "javascript", "react", "angular", "vue",
                "typescript", "next.js", "webpack", "bootstrap", "tailwind", "sass", "es6",
                "responsive design", "web accessibility", "dom", "jquery", "redux",
                "vite", "zustand", "framer motion", "storybook", "eslint", "vitepress", "pwa",
                "single page application", "csr", "ssr", "hydration", "component-based ui",
                "web components", "micro frontends", "bundler", "transpiler", "polyfill", "css grid",
                "flexbox", "css animations", "web performance", "lighthouse", "core web vitals"
            ],
            
            "Backend Development": [
                "backend", "node.js", "django", "flask", "express", "api development",
                "sql", "nosql", "server-side", "mysql", "postgresql", "mongodb", "rest api",
                "graphql", "java", "spring boot", "authentication", "authorization", "mvc",
                "business logic", "orm", "database schema", "asp.net", "laravel", "go", "fastapi",
                "nest.js", "microservices", "websockets", "rabbitmq", "message broker", "cron jobs",
                "redis", "elasticsearch", "kafka", "grpc", "soap", "middleware", "caching",
                "load balancing", "rate limiting", "api gateway", "serverless", "lambda functions"
            ],
            
            "Full Stack Development": [
                "full stack", "fullstack", "mern", "mean", "mevn", "lamp", "jamstack",
                "frontend and backend", "end-to-end development", "full stack developer",
                "api integration", "rest api", "graphql", "react + node", "react.js + express",
                "monolith", "microservices", "serverless architecture", "integrated app",
                "web application", "cross-functional development", "component-based architecture",
                "database design", "middleware", "mvc", "mvvm", "authentication", "authorization",
                "session management", "cloud deployment", "responsive ui", "performance tuning",
                "state management", "redux", "context api", "axios", "fetch api", "isomorphic",
                "universal rendering", "headless cms", "api-first development"
            ],
            
            "Cybersecurity": [
                "cybersecurity", "security analyst", "penetration testing", "ethical hacking",
                "owasp", "vulnerability", "threat analysis", "infosec", "red team", "blue team",
                "incident response", "firewall", "ids", "ips", "malware", "encryption",
                "cyber threat", "security operations", "siem", "zero-day", "cyber attack",
                "kali linux", "burp suite", "nmap", "wireshark", "cve", "forensics",
                "security audit", "information security", "compliance", "ransomware",
                "threat hunting", "security architecture", "identity management", "pki",
                "security governance", "risk assessment", "vulnerability management", "soc"
            ],
            
            "Cloud Engineering": [
                "cloud", "aws", "azure", "gcp", "cloud engineer", "cloud computing",
                "cloud infrastructure", "cloud security", "s3", "ec2", "cloud formation",
                "load balancer", "auto scaling", "cloud storage", "cloud native", "cloud migration",
                "eks", "aks", "terraform", "cloudwatch", "cloudtrail", "iam", "rds", "elb",
                "lambda", "azure functions", "cloud functions", "serverless", "containers",
                "cloud architecture", "multi-cloud", "hybrid cloud", "cloud cost optimization"
            ],
            
            "DevOps/Infrastructure": [
                "devops", "docker", "kubernetes", "ci/cd", "jenkins", "ansible",
                "infrastructure as code", "terraform", "monitoring", "prometheus", "grafana",
                "deployment", "automation", "pipeline", "build and release", "scripting",
                "bash", "shell script", "site reliability", "sre", "argocd", "helm", "fluxcd",
                "aws cli", "linux administration", "log aggregation", "observability", "splunk",
                "gitlab ci", "github actions", "azure devops", "puppet", "chef", "vagrant",
                "infrastructure monitoring", "alerting", "incident management", "chaos engineering"
            ],
            
            "Quality Assurance": [
                "qa", "quality assurance", "testing", "test automation", "selenium", "cypress",
                "test cases", "test planning", "bug tracking", "regression testing", "performance testing",
                "load testing", "stress testing", "api testing", "ui testing", "unit testing",
                "integration testing", "system testing", "acceptance testing", "test driven development",
                "behavior driven development", "cucumber", "jest", "mocha", "junit", "testng",
                "postman", "jmeter", "appium", "test management", "defect management"
            ],
            
            "Game Development": [
                "game development", "unity", "unreal engine", "c#", "c++", "game design",
                "game programming", "3d modeling", "animation", "shader programming", "physics engine",
                "game mechanics", "level design", "game testing", "multiplayer", "networking",
                "mobile games", "console games", "pc games", "vr games", "ar games",
                "game optimization", "performance profiling", "game analytics", "monetization"
            ],
            
            "Blockchain Development": [
                "blockchain", "cryptocurrency", "smart contracts", "solidity", "ethereum",
                "bitcoin", "defi", "nft", "web3", "dapp", "consensus algorithms",
                "cryptography", "distributed ledger", "mining", "staking", "tokenomics",
                "metamask", "truffle", "hardhat", "ipfs", "polygon", "binance smart chain",
                "hyperledger", "chainlink", "oracles", "dao", "yield farming"
            ],
            
            "Embedded Systems": [
                "embedded systems", "microcontroller", "firmware", "c programming", "assembly",
                "real-time systems", "rtos", "arduino", "raspberry pi", "arm", "pic",
                "embedded c", "hardware programming", "sensor integration", "iot devices",
                "low-level programming", "device drivers", "bootloader", "embedded linux",
                "fpga", "verilog", "vhdl", "pcb design", "circuit design"
            ],
            
            "System Architecture": [
                "system architecture", "solution architect", "enterprise architecture", "microservices",
                "distributed systems", "scalability", "high availability", "fault tolerance",
                "system design", "architecture patterns", "design patterns", "load balancing",
                "caching strategies", "database sharding", "event-driven architecture", "message queues",
                "api design", "service mesh", "containerization", "orchestration", "cloud architecture"
            ],
            
            "Database Management": [
                "database administrator", "dba", "database design", "sql optimization",
                "database performance", "backup and recovery", "replication", "clustering",
                "data modeling", "normalization", "indexing", "stored procedures", "triggers",
                "database security", "mysql", "postgresql", "oracle", "sql server", "mongodb",
                "cassandra", "redis", "elasticsearch", "data warehouse", "etl", "olap"
            ],
            
            "Networking": [
                "network engineer", "network administration", "cisco", "routing", "switching",
                "tcp/ip", "dns", "dhcp", "vpn", "firewall", "network security",
                "network monitoring", "network troubleshooting", "wan", "lan", "vlan",
                "bgp", "ospf", "mpls", "sd-wan", "network automation", "network protocols"
            ],
            
            "Site Reliability Engineering": [
                "sre", "site reliability", "system reliability", "incident management",
                "post-mortem", "error budgets", "sli", "slo", "monitoring", "alerting",
                "capacity planning", "performance optimization", "chaos engineering",
                "disaster recovery", "high availability", "fault tolerance", "observability"
            ],
            
            "Product Management": [
                "product manager", "product management", "product strategy", "roadmap",
                "user stories", "requirements gathering", "stakeholder management", "agile",
                "scrum", "kanban", "product analytics", "a/b testing", "user research",
                "market research", "competitive analysis", "go-to-market", "product launch",
                "feature prioritization", "backlog management", "kpi", "metrics"
            ],
            
            "Project Management": [
                "project manager", "project management", "pmp", "agile", "scrum master",
                "kanban", "waterfall", "risk management", "resource planning", "timeline",
                "milestone", "deliverables", "stakeholder communication", "budget management",
                "team coordination", "project planning", "project execution", "project closure",
                "change management", "quality assurance", "jira", "confluence", "ms project"
            ],
            
            "Business Analysis": [
                "business analyst", "requirements analysis", "process improvement", "workflow",
                "business process", "stakeholder analysis", "gap analysis", "use cases",
                "functional requirements", "non-functional requirements", "documentation",
                "process mapping", "business rules", "acceptance criteria", "user acceptance testing",
                "change management", "business intelligence", "data analysis", "reporting"
            ],
            
            "Technical Writing": [
                "technical writer", "documentation", "api documentation", "user manuals",
                "technical communication", "content strategy", "information architecture",
                "style guide", "editing", "proofreading", "markdown", "confluence",
                "gitbook", "sphinx", "doxygen", "technical blogging", "knowledge base"
            ],
            
            "Digital Marketing": [
                "digital marketing", "seo", "sem", "social media marketing", "content marketing",
                "email marketing", "ppc", "google ads", "facebook ads", "analytics",
                "conversion optimization", "marketing automation", "lead generation",
                "brand management", "influencer marketing", "affiliate marketing", "growth hacking"
            ],
            
            "E-commerce": [
                "e-commerce", "online retail", "shopify", "magento", "woocommerce",
                "payment gateway", "inventory management", "order management", "shipping",
                "customer service", "marketplace", "dropshipping", "conversion rate optimization",
                "product catalog", "shopping cart", "checkout optimization", "amazon fba"
            ],
            
            "Fintech": [
                "fintech", "financial technology", "payment processing", "banking software",
                "trading systems", "risk management", "compliance", "regulatory", "kyc",
                "aml", "blockchain finance", "cryptocurrency", "robo-advisor", "insurtech",
                "lending platform", "credit scoring", "fraud detection", "financial analytics"
            ],
            
            "Healthcare Tech": [
                "healthcare technology", "healthtech", "medical software", "ehr", "emr",
                "telemedicine", "medical devices", "hipaa", "healthcare analytics",
                "clinical trials", "medical imaging", "bioinformatics", "health informatics",
                "patient management", "healthcare compliance", "medical ai", "digital health"
            ],
            
            "EdTech": [
                "edtech", "educational technology", "e-learning", "lms", "learning management",
                "online education", "educational software", "student information system",
                "assessment tools", "educational analytics", "adaptive learning", "gamification",
                "virtual classroom", "educational content", "curriculum development"
            ],
            
            "IoT Development": [
                "iot", "internet of things", "connected devices", "sensor networks",
                "edge computing", "mqtt", "coap", "zigbee", "bluetooth", "wifi",
                "embedded systems", "device management", "iot platform", "industrial iot",
                "smart home", "smart city", "wearables", "asset tracking", "predictive maintenance"
            ],
            
            "AR/VR Development": [
                "ar", "vr", "augmented reality", "virtual reality", "mixed reality", "xr",
                "unity 3d", "unreal engine", "oculus", "hololens", "arkit", "arcore",
                "3d modeling", "spatial computing", "immersive experience", "360 video",
                "haptic feedback", "motion tracking", "computer vision", "3d graphics"
            ],
            
            "Technical Sales": [
                "technical sales", "sales engineer", "solution selling", "pre-sales",
                "technical consulting", "customer success", "account management",
                "product demonstration", "technical presentation", "proposal writing",
                "client relationship", "revenue generation", "sales process", "crm"
            ],
            
            "Agile Coaching": [
                "agile coach", "scrum master", "agile transformation", "team facilitation",
                "retrospectives", "sprint planning", "daily standups", "agile ceremonies",
                "continuous improvement", "change management", "team dynamics",
                "agile metrics", "coaching", "mentoring", "organizational change"
            ],
            
            "Software Engineering": [
                "software engineer", "web developer", "developer", "programmer",
                "object oriented", "design patterns", "agile", "scrum", "git", "version control",
                "unit testing", "integration testing", "debugging", "code review", "system design",
                "tdd", "bdd", "pair programming", "refactoring", "uml", "dev environment", "ide",
                "algorithms", "data structures", "software architecture", "clean code"
            ]
        }

        # Step 1: Compute weighted keyword matches (4x for title, 1x for desc)
        for domain, kws in keywords.items():
            title_hits = sum(1 for kw in kws if kw in title)
            desc_hits = sum(1 for kw in kws if kw in desc)
            domain_scores[domain] = (4 * title_hits + 1 * desc_hits) * WEIGHTS[domain]

        # Step 2: Enhanced Full Stack Detection
        frontend_hits = sum(1 for kw in keywords["Frontend Development"] if kw in title or kw in desc)
        backend_hits = sum(1 for kw in keywords["Backend Development"] if kw in title or kw in desc)
        fullstack_mentioned = any(term in title or term in desc for term in ["full stack", "fullstack", "full-stack"])

        if fullstack_mentioned:
            domain_scores["Full Stack Development"] += 15

        if frontend_hits >= 4 and backend_hits >= 4:
            domain_scores["Full Stack Development"] += 12

        # Step 3: Domain-specific boosts
        domain_boosts = {
            "AI/Machine Learning": ["ai", "ml", "machine learning", "artificial intelligence"],
            "Cybersecurity": ["security", "cyber", "infosec"],
            "Cloud Engineering": ["cloud", "aws", "azure", "gcp"],
            "Mobile Development": ["mobile", "android", "ios", "app"],
            "Game Development": ["game", "unity", "unreal"],
            "Blockchain Development": ["blockchain", "crypto", "web3", "defi"],
            "IoT Development": ["iot", "embedded", "sensor"],
            "AR/VR Development": ["ar", "vr", "augmented", "virtual reality"]
        }

        for domain, boost_terms in domain_boosts.items():
            if any(term in title for term in boost_terms):
                domain_scores[domain] += 8
            if any(term in desc for term in boost_terms):
                domain_scores[domain] += 3

        # Step 4: Filter short/noisy descriptions
        if len(desc.split()) < 8:
            for domain in domain_scores:
                desc_hits = sum(1 for kw in keywords[domain] if kw in desc)
                domain_scores[domain] = max(0, domain_scores[domain] - (desc_hits * WEIGHTS[domain] * 0.5))

        # Step 5: Choose top domain
        if domain_scores:
            top_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[top_domain] > 0:
                return top_domain

        return "Software Engineering"

    def get_domain_similarity(self, resume_domain: str, job_domain: str) -> float:
        """Enhanced similarity scoring with comprehensive domain relationships"""
        
        resume_domain = resume_domain.strip().lower()
        job_domain = job_domain.strip().lower()

        # Enhanced normalization
        normalization = {
            "frontend": "frontend development",
            "backend": "backend development",
            "fullstack": "full stack development",
            "full-stack": "full stack development",
            "ui/ux": "ui/ux design",
            "ux/ui": "ui/ux design",
            "software developer": "software engineering",
            "mobile developer": "mobile development",
            "android developer": "mobile development",
            "ios developer": "mobile development",
            "ai": "ai/machine learning",
            "machine learning": "ai/machine learning",
            "ml": "ai/machine learning",
            "artificial intelligence": "ai/machine learning",
            "cloud": "cloud engineering",
            "cloud engineer": "cloud engineering",
            "devops": "devops/infrastructure",
            "devops engineer": "devops/infrastructure",
            "cyber security": "cybersecurity",
            "cybersecurity engineer": "cybersecurity",
            "security analyst": "cybersecurity",
            "qa": "quality assurance",
            "test engineer": "quality assurance",
            "sre": "site reliability engineering",
            "dba": "database management",
            "database administrator": "database management",
            "product manager": "product management",
            "project manager": "project management",
            "business analyst": "business analysis",
            "technical writer": "technical writing",
            "game developer": "game development",
            "blockchain developer": "blockchain development"
        }

        resume_domain = normalization.get(resume_domain, resume_domain)
        job_domain = normalization.get(job_domain, job_domain)

        # Comprehensive similarity mapping with detailed relationships
        similarity_map = {
            # Full Stack relationships
            ("full stack development", "frontend development"): 0.85,
            ("full stack development", "backend development"): 0.85,
            ("full stack development", "ui/ux design"): 0.70,
            ("full stack development", "mobile development"): 0.65,
            ("full stack development", "software engineering"): 0.80,
            
            # Frontend relationships
            ("frontend development", "ui/ux design"): 0.90,
            ("frontend development", "mobile development"): 0.70,
            ("frontend development", "software engineering"): 0.75,
            ("frontend development", "backend development"): 0.60,
            
            # Backend relationships
            ("backend development", "database management"): 0.80,
            ("backend development", "cloud engineering"): 0.75,
            ("backend development", "devops/infrastructure"): 0.70,
            ("backend development", "system architecture"): 0.85,
            ("backend development", "software engineering"): 0.80,
            
            # Data & AI relationships
            ("data science", "ai/machine learning"): 0.95,
            ("data science", "business analysis"): 0.70,
            ("ai/machine learning", "data science"): 0.95,
            ("ai/machine learning", "software engineering"): 0.65,
            
            # Cloud & Infrastructure relationships
            ("cloud engineering", "devops/infrastructure"): 0.90,
            ("cloud engineering", "system architecture"): 0.80,
            ("cloud engineering", "site reliability engineering"): 0.85,
            ("devops/infrastructure", "site reliability engineering"): 0.90,
            ("devops/infrastructure", "system architecture"): 0.75,
            
            # Security relationships
            ("cybersecurity", "devops/infrastructure"): 0.70,
            ("cybersecurity", "cloud engineering"): 0.75,
            ("cybersecurity", "networking"): 0.80,
            ("cybersecurity", "system architecture"): 0.65,
            
            # Mobile relationships
            ("mobile development", "ui/ux design"): 0.75,
            ("mobile development", "software engineering"): 0.70,
            ("mobile development", "game development"): 0.60,
            
            # Quality & Testing relationships
            ("quality assurance", "software engineering"): 0.75,
            ("quality assurance", "devops/infrastructure"): 0.65,
            ("quality assurance", "system architecture"): 0.60,
            
            # Management relationships
            ("product management", "business analysis"): 0.80,
            ("product management", "project management"): 0.75,
            ("project management", "agile coaching"): 0.85,
            ("business analysis", "data science"): 0.65,
            
            # Specialized tech relationships
            ("game development", "software engineering"): 0.70,
            ("blockchain development", "software engineering"): 0.70,
            ("blockchain development", "cybersecurity"): 0.65,
            ("embedded systems", "iot development"): 0.90,
            ("ar/vr development", "game development"): 0.80,
            ("ar/vr development", "mobile development"): 0.70,
            
            # Database relationships
            ("database management", "data science"): 0.75,
            ("database management", "system architecture"): 0.70,
            ("database management", "backend development"): 0.80,
            
            # Architecture relationships
            ("system architecture", "software engineering"): 0.85,
            ("system architecture", "cloud engineering"): 0.80,
            ("system architecture", "backend development"): 0.85,
            
            # Networking relationships
            ("networking", "cybersecurity"): 0.80,
            ("networking", "devops/infrastructure"): 0.75,
            ("networking", "system architecture"): 0.70,
            
            # Industry-specific relationships
            ("fintech", "software engineering"): 0.70,
            ("fintech", "backend development"): 0.75,
            ("fintech", "cybersecurity"): 0.70,
            ("healthcare tech", "software engineering"): 0.70,
            ("edtech", "software engineering"): 0.70,
            ("e-commerce", "full stack development"): 0.80,
            ("e-commerce", "backend development"): 0.75,
            
            # Sales & Communication relationships
            ("technical sales", "product management"): 0.65,
            ("technical writing", "business analysis"): 0.60,
            ("digital marketing", "business analysis"): 0.55,
            
            # General software relationships
            ("software engineering", "full stack development"): 0.80,
            ("software engineering", "frontend development"): 0.75,
            ("software engineering", "backend development"): 0.80,
            ("software engineering", "mobile development"): 0.70,
            ("software engineering", "game development"): 0.70,
            ("software engineering", "quality assurance"): 0.75,
        }

        # Perfect match
        if resume_domain == job_domain:
            return 1.0

        # Check similarity map (bidirectional)
        similarity = (similarity_map.get((resume_domain, job_domain)) or 
                     similarity_map.get((job_domain, resume_domain)))
        
        if similarity:
            return similarity

        # Enhanced fallback logic for related domains
        tech_domains = {
            "software engineering", "full stack development", "frontend development", 
            "backend development", "mobile development", "game development", 
            "blockchain development", "embedded systems", "iot development"
        }
        
        data_domains = {
            "data science", "ai/machine learning", "business analysis"
        }
        
        infrastructure_domains = {
            "cloud engineering", "devops/infrastructure", "site reliability engineering",
            "system architecture", "database management", "networking", "cybersecurity"
        }
        
        management_domains = {
            "product management", "project management", "business analysis", "agile coaching"
        }
        
        design_domains = {
            "ui/ux design", "ar/vr development"
        }

        # Same category bonus
        categories = [tech_domains, data_domains, infrastructure_domains, management_domains, design_domains]
        for category in categories:
            if resume_domain in category and job_domain in category:
                return 0.50  # Moderate similarity for same category
        
        # Cross-category relationships
        if ((resume_domain in tech_domains and job_domain in infrastructure_domains) or
            (resume_domain in infrastructure_domains and job_domain in tech_domains)):
            return 0.45
        
        if ((resume_domain in data_domains and job_domain in tech_domains) or
            (resume_domain in tech_domains and job_domain in data_domains)):
            return 0.40

        # Default low similarity for unrelated domains
        return 0.25

    def insert_candidate(self, data: Tuple, job_title: str = "", job_description: str = "") -> int:
        """
        Enhanced insert function with better domain handling and error checking
        Returns the ID of the inserted candidate
        """
        try:
            local_tz = pytz.timezone("Asia/Kolkata")
            local_time = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")

            # Detect domain from job title + description
            detected_domain = self.detect_domain_from_title_and_description(job_title, job_description)

            # Validate data length and types
            if len(data) < 9:
                raise ValueError(f"Expected at least 9 data fields, got {len(data)}")

            # Use only first 9 values and append domain
            normalized_data = data[:9] + (detected_domain,)

            # Validate score ranges
            for i, score in enumerate(normalized_data[2:8]):  # ats_score to keyword_score
                if not isinstance(score, (int, float)) or not (0 <= score <= 100):
                    raise ValueError(f"Score at position {i+2} must be between 0 and 100, got {score}")

            # Validate bias score
            bias_score = normalized_data[8]
            if not isinstance(bias_score, (int, float)) or not (0.0 <= bias_score <= 1.0):
                raise ValueError(f"Bias score must be between 0.0 and 1.0, got {bias_score}")

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO candidates (
                        resume_name, candidate_name, ats_score, edu_score, exp_score,
                        skills_score, lang_score, keyword_score, bias_score, domain, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, normalized_data + (local_time,))
                conn.commit()
                candidate_id = cursor.lastrowid
                logger.info(f"Inserted candidate with ID: {candidate_id}")
                return candidate_id

        except Exception as e:
            logger.error(f"Error inserting candidate: {e}")
            raise

    def get_top_domains_by_score(self, limit: int = 5) -> List[Tuple]:
        """Get top domains by ATS score with optimized query"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT domain, ROUND(AVG(ats_score), 2) AS avg_score, COUNT(*) AS count
                    FROM candidates
                    GROUP BY domain
                    HAVING count >= 1
                    ORDER BY avg_score DESC
                    LIMIT ?
                """, (limit,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting top domains: {e}")
            return []

    def get_resume_count_by_day(self) -> pd.DataFrame:
        """Resume count by date with optimized query"""
        try:
            query = """
                SELECT DATE(timestamp) AS day, COUNT(*) AS count
                FROM candidates
                GROUP BY DATE(timestamp)
                ORDER BY DATE(timestamp) DESC
                LIMIT 365
            """
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error getting resume count by day: {e}")
            return pd.DataFrame()

    def get_average_ats_by_domain(self) -> pd.DataFrame:
        """Average ATS score by domain with optimized query"""
        try:
            query = """
                SELECT domain, 
                       ROUND(AVG(ats_score), 2) AS avg_ats_score,
                       COUNT(*) as candidate_count
                FROM candidates
                GROUP BY domain
                HAVING candidate_count >= 1
                ORDER BY avg_ats_score DESC
            """
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error getting average ATS by domain: {e}")
            return pd.DataFrame()

    def get_domain_distribution(self) -> pd.DataFrame:
        """Resume distribution by domain with percentage calculation"""
        try:
            query = """
                SELECT domain, 
                       COUNT(*) as count,
                       ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM candidates), 2) as percentage
                FROM candidates
                GROUP BY domain
                ORDER BY count DESC
            """
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error getting domain distribution: {e}")
            return pd.DataFrame()

    def filter_candidates_by_date(self, start: str, end: str) -> pd.DataFrame:
        """Filter candidates by date range with validation"""
        try:
            # Validate date format
            datetime.strptime(start, '%Y-%m-%d')
            datetime.strptime(end, '%Y-%m-%d')
            
            query = """
                SELECT * FROM candidates
                WHERE DATE(timestamp) BETWEEN DATE(?) AND DATE(?)
                ORDER BY timestamp DESC
            """
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn, params=(start, end))
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error filtering candidates by date: {e}")
            return pd.DataFrame()

    def delete_candidate_by_id(self, candidate_id: int) -> bool:
        """Delete candidate by ID with validation"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Deleted candidate with ID: {candidate_id}")
                    return True
                else:
                    logger.warning(f"No candidate found with ID: {candidate_id}")
                    return False
        except Exception as e:
            logger.error(f"Error deleting candidate: {e}")
            return False

    def get_all_candidates(self, bias_threshold: Optional[float] = None, 
                          min_ats: Optional[int] = None, 
                          limit: Optional[int] = None,
                          offset: int = 0) -> pd.DataFrame:
        """Get all candidates with optional filters and pagination"""
        try:
            query = "SELECT * FROM candidates WHERE 1=1"
            params = []

            if bias_threshold is not None:
                query += " AND bias_score >= ?"
                params.append(bias_threshold)

            if min_ats is not None:
                query += " AND ats_score >= ?"
                params.append(min_ats)

            query += " ORDER BY timestamp DESC"
            
            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"Error getting all candidates: {e}")
            return pd.DataFrame()

    def export_to_csv(self, filepath: str = "candidates_export.csv", 
                     filters: Optional[Dict[str, Any]] = None) -> bool:
        """Export candidate data to CSV with optional filters"""
        try:
            query = "SELECT * FROM candidates WHERE 1=1"
            params = []
            
            if filters:
                if 'min_ats' in filters:
                    query += " AND ats_score >= ?"
                    params.append(filters['min_ats'])
                if 'domain' in filters:
                    query += " AND domain = ?"
                    params.append(filters['domain'])
                if 'start_date' in filters:
                    query += " AND DATE(timestamp) >= DATE(?)"
                    params.append(filters['start_date'])
                if 'end_date' in filters:
                    query += " AND DATE(timestamp) <= DATE(?)"
                    params.append(filters['end_date'])
            
            query += " ORDER BY timestamp DESC"
            
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                df.to_csv(filepath, index=False)
                logger.info(f"Exported {len(df)} records to {filepath}")
                return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

    def get_candidate_by_id(self, candidate_id: int) -> pd.DataFrame:
        """Get a specific candidate by ID"""
        try:
            query = "SELECT * FROM candidates WHERE id = ?"
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn, params=(candidate_id,))
        except Exception as e:
            logger.error(f"Error getting candidate by ID: {e}")
            return pd.DataFrame()

    def get_bias_distribution(self, threshold: float = 0.6) -> pd.DataFrame:
        """Get bias score distribution with validation"""
        try:
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("Threshold must be between 0.0 and 1.0")
                
            query = """
                SELECT 
                    CASE WHEN bias_score >= ? THEN 'Biased' ELSE 'Fair' END AS bias_category,
                    COUNT(*) AS count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM candidates), 2) as percentage
                FROM candidates
                GROUP BY bias_category
            """
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn, params=(threshold,))
        except Exception as e:
            logger.error(f"Error getting bias distribution: {e}")
            return pd.DataFrame()

    def get_daily_ats_stats(self, days_limit: int = 90) -> pd.DataFrame:
        """ATS score trend over time with limit"""
        try:
            query = """
                SELECT DATE(timestamp) AS date, 
                       ROUND(AVG(ats_score), 2) AS avg_ats,
                       COUNT(*) as daily_count
                FROM candidates
                WHERE DATE(timestamp) >= DATE('now', '-{} days')
                GROUP BY DATE(timestamp)
                ORDER BY DATE(timestamp)
            """.format(days_limit)
            
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error getting daily ATS stats: {e}")
            return pd.DataFrame()

    def get_flagged_candidates(self, threshold: float = 0.6) -> pd.DataFrame:
        """Get all flagged candidates with validation"""
        try:
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("Threshold must be between 0.0 and 1.0")
                
            query = """
                SELECT resume_name, candidate_name, ats_score, bias_score, domain, timestamp
                FROM candidates
                WHERE bias_score > ?
                ORDER BY bias_score DESC
            """
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn, params=(threshold,))
        except Exception as e:
            logger.error(f"Error getting flagged candidates: {e}")
            return pd.DataFrame()

    def get_domain_performance_stats(self) -> pd.DataFrame:
        """Get comprehensive domain performance statistics"""
        try:
            query = """
                SELECT 
                    domain,
                    COUNT(*) as total_candidates,
                    ROUND(AVG(ats_score), 2) as avg_ats_score,
                    ROUND(AVG(edu_score), 2) as avg_edu_score,
                    ROUND(AVG(exp_score), 2) as avg_exp_score,
                    ROUND(AVG(skills_score), 2) as avg_skills_score,
                    ROUND(AVG(lang_score), 2) as avg_lang_score,
                    ROUND(AVG(keyword_score), 2) as avg_keyword_score,
                    ROUND(AVG(bias_score), 3) as avg_bias_score,
                    MAX(ats_score) as max_ats_score,
                    MIN(ats_score) as min_ats_score,
                    ROUND(MAX(ats_score) - MIN(ats_score), 2) as score_range
                FROM candidates
                GROUP BY domain
                HAVING total_candidates >= 1
                ORDER BY avg_ats_score DESC
            """
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error getting domain performance stats: {e}")
            return pd.DataFrame()

    def analyze_domain_transitions(self) -> pd.DataFrame:
        """Analyze domain frequency and performance"""
        try:
            query = """
                SELECT 
                    domain,
                    COUNT(*) as frequency,
                    ROUND(AVG(ats_score), 2) as avg_performance,
                    ROUND(AVG(bias_score), 3) as avg_bias,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM candidates), 2) as percentage
                FROM candidates
                GROUP BY domain
                HAVING frequency >= 1
                ORDER BY frequency DESC
            """
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error analyzing domain transitions: {e}")
            return pd.DataFrame()

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Total candidates
                cursor.execute("SELECT COUNT(*) FROM candidates")
                total_candidates = cursor.fetchone()[0]
                
                # Average scores
                cursor.execute("""
                    SELECT 
                        ROUND(AVG(ats_score), 2) as avg_ats,
                        ROUND(AVG(bias_score), 3) as avg_bias,
                        COUNT(DISTINCT domain) as unique_domains
                    FROM candidates
                """)
                avg_stats = cursor.fetchone()
                
                # Date range
                cursor.execute("""
                    SELECT 
                        MIN(DATE(timestamp)) as earliest_date,
                        MAX(DATE(timestamp)) as latest_date
                    FROM candidates
                """)
                date_range = cursor.fetchone()
                
                return {
                    'total_candidates': total_candidates,
                    'avg_ats_score': avg_stats[0] if avg_stats[0] else 0,
                    'avg_bias_score': avg_stats[1] if avg_stats[1] else 0,
                    'unique_domains': avg_stats[2] if avg_stats[2] else 0,
                    'earliest_date': date_range[0],
                    'latest_date': date_range[1],
                    'database_size_mb': round(os.path.getsize(self.db_path) / (1024 * 1024), 2) if os.path.exists(self.db_path) else 0
                }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

    def cleanup_old_records(self, days_to_keep: int = 365) -> int:
        """Clean up old records beyond specified days"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM candidates 
                    WHERE DATE(timestamp) < DATE('now', '-{} days')
                """.format(days_to_keep))
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old records")
                    # Vacuum to reclaim space
                    cursor.execute("VACUUM")
                    
                return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")
            return 0

    def close_all_connections(self):
        """Close all connections in the pool"""
        with self._pool_lock:
            while self._connection_pool:
                conn = self._connection_pool.pop()
                conn.close()
            logger.info("All database connections closed")


# Create global instance for backward compatibility
db_manager = DatabaseManager()

# Export functions for backward compatibility
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

# Additional utility functions
def get_database_stats():
    return db_manager.get_database_stats()

def cleanup_old_records(days_to_keep: int = 365):
    return db_manager.cleanup_old_records(days_to_keep)

def close_all_connections():
    return db_manager.close_all_connections()

if __name__ == "__main__":
    # Example usage and testing
    print("Database Manager initialized successfully!")
    stats = get_database_stats()
    print(f"Database Statistics: {stats}")

# Initialize a global instance of DatabaseManager
db = DatabaseManager()
