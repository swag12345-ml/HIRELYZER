"""
Enhanced Database Manager for Resume Analysis System
Optimized for large-scale user structures with improved performance and reliability
"""

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

    def detect_domain_from_title_and_description(self, job_title: str, job_description: str) -> str:
        """
        Enhanced Domain Detection with 25+ Professional Domains
        Optimized for better performance with cached keyword lookups and confidence thresholding
        """
        title = job_title.lower().strip()
        desc = job_description.lower().strip()

        # Enhanced normalization with more synonyms
        replacements = {
            "cyber security": "cybersecurity",
            "ai engineer": "machine learning",
            "ml engineer": "machine learning",
            "software developer": "software engineer",
            "programmer": "software engineer",
            "coder": "software engineer",
            "web developer": "software engineer",
            "frontend developer": "frontend",
            "front-end developer": "frontend",
            "front end developer": "frontend",
            "backend developer": "backend",
            "back-end developer": "backend", 
            "back end developer": "backend",
            "fullstack developer": "full stack",
            "full-stack developer": "full stack",
            "full stack developer": "full stack",
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
            "solution architect": "system architecture",
            "mobile developer": "mobile development",
            "android developer": "mobile development",
            "ios developer": "mobile development",
            "react native developer": "mobile development"
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

        # Comprehensive and expanded keyword mapping for 30+ domains
        keywords = {
            "Data Science": [
                "data analyst", "data scientist", "data science", "eda", "pandas", "numpy",
                "data analysis", "statistics", "data visualization", "matplotlib", "seaborn", "plotly",
                "power bi", "tableau", "looker", "qlik", "kpi", "sql", "excel", "dashboards",
                "insights", "hypothesis testing", "a/b testing", "business intelligence", "data wrangling",
                "feature engineering", "data storytelling", "exploratory analysis", "data mining",
                "statistical modeling", "time series", "forecasting", "predictive analytics", "analytics engineer",
                "r programming", "jupyter", "databricks", "spark", "hadoop", "etl", "data pipeline",
                "data warehouse", "olap", "oltp", "dimensional modeling", "data governance",
                "data cleaning", "data preprocessing", "statistical analysis", "regression analysis",
                "correlation", "variance", "standard deviation", "confidence intervals", "p-values",
                "data lakes", "big data", "apache spark", "snowflake", "redshift", "bigquery"
            ],
            
            "AI/Machine Learning": [
                "machine learning", "ml engineer", "deep learning", "neural network", "artificial intelligence",
                "nlp", "natural language processing", "computer vision", "ai engineer", "scikit-learn", "tensorflow", "pytorch",
                "llm", "large language model", "huggingface", "xgboost", "lightgbm", "classification", "regression",
                "reinforcement learning", "transfer learning", "model training", "bert", "gpt", "chatgpt",
                "yolo", "transformer", "autoencoder", "ai models", "fine-tuning", "zero-shot", "one-shot", "few-shot",
                "mistral", "llama", "openai", "langchain", "vector embeddings", "prompt engineering",
                "mlops", "model deployment", "feature store", "model monitoring", "hyperparameter tuning",
                "ensemble methods", "gradient boosting", "random forest", "svm", "support vector machine", "clustering", "pca",
                "keras", "opencv", "spacy", "nltk", "gensim", "word2vec", "glove", "attention mechanism",
                "generative ai", "gan", "diffusion models", "stable diffusion", "image generation", "text generation",
                "recommendation systems", "collaborative filtering", "content-based filtering", "matrix factorization"
            ],
            
            "UI/UX Design": [
                "figma", "adobe xd", "sketch", "wireframe", "prototyping", "mockup", "design system",
                "user interface", "user experience", "usability testing", "user testing", "ux research",
                "interaction design", "visual design", "graphic design", "responsive design", 
                "material design", "human interface guidelines", "user research", "personas", "user journey",
                "usability", "accessibility", "wcag", "human-centered design", "design thinking",
                "affinity diagram", "journey mapping", "heuristic evaluation", "card sorting",
                "mobile-first", "ux audit", "design tokens", "atomic design", "design ops",
                "information architecture", "tree testing", "a/b testing design", "design sprint", 
                "brand design", "typography", "color theory", "layout design", "ui components",
                "design patterns", "user flows", "site mapping", "content strategy", "design systems"
            ],
            
            "Mobile Development": [
                "android", "ios", "flutter", "kotlin", "swift", "mobile app", "react native", "xamarin",
                "mobile application", "play store", "app store", "firebase", "mobile sdk", "cordova",
                "xcode", "android studio", "cross-platform", "native mobile", "push notifications",
                "in-app purchases", "mobile ui", "mobile ux", "apk", "ipa", "expo", "capacitor", "phonegap",
                "ionic", "mobile testing", "app optimization", "mobile security", "app development",
                "offline functionality", "mobile analytics", "app monetization", "mobile performance",
                "objective-c", "java android", "android jetpack", "swiftui", "uikit", "core data",
                "realm", "sqlite mobile", "mobile ci/cd", "app distribution", "mobile debugging",
                "mobile frameworks", "hybrid apps", "progressive web apps", "pwa", "mobile first design"
            ],
            
            "Frontend Development": [
                "frontend", "front-end", "html", "css", "javascript", "react", "angular", "vue", "svelte",
                "typescript", "next.js", "nuxt.js", "webpack", "vite", "bootstrap", "tailwind", "sass", "less", "es6",
                "responsive design", "web accessibility", "dom", "jquery", "redux", "mobx", "vuex", "pinia",
                "zustand", "framer motion", "storybook", "eslint", "prettier", "babel", "pwa",
                "single page application", "spa", "csr", "ssr", "server-side rendering", "hydration", "component-based ui",
                "web components", "micro frontends", "bundler", "transpiler", "polyfill", "css grid",
                "flexbox", "css animations", "web performance", "lighthouse", "core web vitals", "seo",
                "web assembly", "wasm", "graphql client", "apollo client", "relay", "state management",
                "css preprocessor", "css modules", "styled components", "emotion", "material-ui", "chakra ui"
            ],
            
            "Backend Development": [
                "backend", "back-end", "server-side", "node.js", "django", "flask", "fastapi", "express", "api development",
                "sql", "nosql", "mysql", "postgresql", "mongodb", "redis", "rest api", "restful",
                "graphql", "java", "spring boot", "spring framework", "authentication", "authorization", "jwt", "oauth", "mvc",
                "business logic", "orm", "database schema", "asp.net", "laravel", "php", "go", "golang", "rust",
                "nest.js", "microservices", "websockets", "socket.io", "rabbitmq", "message broker", "cron jobs",
                "elasticsearch", "kafka", "apache kafka", "grpc", "soap", "middleware", "caching", "memcached",
                "load balancing", "rate limiting", "api gateway", "serverless", "lambda functions", "azure functions",
                "database design", "stored procedures", "triggers", "indexing", "query optimization",
                "api documentation", "swagger", "openapi", "postman", "insomnia", "database migrations"
            ],
            
            "Full Stack Development": [
                "full stack", "fullstack", "full-stack", "mern", "mean", "mevn", "lamp", "jamstack",
                "frontend and backend", "end-to-end development", "full stack developer", "web development",
                "api integration", "rest api", "graphql", "react + node", "react.js + express", "vue + node",
                "monolith", "microservices", "serverless architecture", "integrated app", "complete solution",
                "web application", "cross-functional development", "component-based architecture",
                "database design", "middleware", "mvc", "mvvm", "authentication", "authorization",
                "session management", "cloud deployment", "responsive ui", "performance tuning",
                "state management", "redux", "context api", "axios", "fetch api", "isomorphic",
                "universal rendering", "headless cms", "api-first development", "javascript stack",
                "python stack", "ruby on rails", "django + react", "laravel + vue", "spring + angular"
            ],
            
            "Cybersecurity": [
                "cybersecurity", "cyber security", "information security", "infosec", "security analyst", "penetration testing", "pen test", "ethical hacking",
                "owasp", "vulnerability", "vulnerability assessment", "threat analysis", "security audit", "red team", "blue team",
                "incident response", "forensics", "digital forensics", "firewall", "ids", "ips", "intrusion detection", "malware", "encryption",
                "cyber threat", "threat hunting", "security operations", "soc", "security operations center", "siem", "zero-day", "cyber attack",
                "kali linux", "burp suite", "nmap", "wireshark", "metasploit", "cve", "security testing",
                "compliance", "ransomware", "phishing", "social engineering", "risk assessment",
                "security architecture", "identity management", "iam", "pki", "public key infrastructure",
                "security governance", "vulnerability management", "patch management", "endpoint security",
                "network security", "application security", "web application security", "secure coding"
            ],
            
            "Cloud Engineering": [
                "cloud", "cloud computing", "aws", "amazon web services", "azure", "microsoft azure", "gcp", "google cloud platform", "cloud engineer",
                "cloud infrastructure", "cloud security", "cloud architecture", "s3", "ec2", "lambda", "azure functions", "cloud functions",
                "load balancer", "auto scaling", "cloud storage", "cloud native", "cloud migration",
                "eks", "aks", "gke", "kubernetes", "docker", "containers", "terraform", "cloudformation", "arm templates",
                "cloudwatch", "azure monitor", "stackdriver", "cloudtrail", "iam", "rds", "cosmos db", "cloud sql", "elb",
                "serverless", "faas", "paas", "iaas", "saas", "multi-cloud", "hybrid cloud", 
                "cloud cost optimization", "cloud governance", "cloud compliance", "cloud backup", "disaster recovery",
                "cdn", "content delivery network", "vpc", "virtual private cloud", "cloud networking"
            ],
            
            "DevOps/Infrastructure": [
                "devops", "dev ops", "infrastructure", "docker", "kubernetes", "k8s", "ci/cd", "continuous integration", "continuous deployment",
                "jenkins", "gitlab ci", "github actions", "azure devops", "ansible", "puppet", "chef", "vagrant",
                "infrastructure as code", "iac", "terraform", "monitoring", "prometheus", "grafana", "elk stack",
                "deployment", "automation", "pipeline", "build and release", "scripting", "orchestration",
                "bash", "shell script", "powershell", "site reliability", "sre", "argocd", "helm", "fluxcd",
                "aws cli", "azure cli", "gcloud", "linux administration", "log aggregation", "observability", "splunk",
                "infrastructure monitoring", "alerting", "incident management", "chaos engineering", "configuration management",
                "containerization", "microservices", "service mesh", "istio", "consul", "vault", "secrets management"
            ],
            
            "Quality Assurance": [
                "qa", "quality assurance", "testing", "software testing", "test automation", "automated testing", "selenium", "cypress", "playwright",
                "test cases", "test planning", "test strategy", "bug tracking", "defect management", "regression testing", "performance testing",
                "load testing", "stress testing", "api testing", "ui testing", "unit testing", "integration testing",
                "integration testing", "system testing", "acceptance testing", "user acceptance testing", "uat", "test driven development", "tdd",
                "behavior driven development", "bdd", "cucumber", "gherkin", "jest", "mocha", "jasmine", "junit", "testng",
                "postman", "insomnia", "jmeter", "k6", "appium", "test management", "test execution",
                "manual testing", "exploratory testing", "black box testing", "white box testing", "gray box testing",
                "accessibility testing", "usability testing", "security testing", "compatibility testing"
            ],
            
            "Game Development": [
                "game development", "game programming", "unity", "unity3d", "unreal engine", "unreal", "c#", "c++", "game design",
                "3d modeling", "animation", "rigging", "shader programming", "shaders", "physics engine", "game physics",
                "game mechanics", "gameplay", "level design", "game testing", "multiplayer", "networking", "game networking",
                "mobile games", "console games", "pc games", "vr games", "ar games", "steam", "playstation", "xbox",
                "game optimization", "performance profiling", "game analytics", "monetization", "in-app purchases",
                "blender", "maya", "3ds max", "substance painter", "texture art", "concept art", "game art",
                "godot", "cocos2d", "construct", "gamemaker", "rpg maker", "indie games", "aaa games"
            ],
            
            "Blockchain Development": [
                "blockchain", "cryptocurrency", "crypto", "smart contracts", "solidity", "ethereum", "web3",
                "bitcoin", "btc", "defi", "decentralized finance", "nft", "non-fungible token", "dapp", "decentralized app", "consensus algorithms",
                "cryptography", "distributed ledger", "mining", "staking", "tokenomics", "token economics",
                "metamask", "truffle", "hardhat", "ganache", "ipfs", "polygon", "matic", "binance smart chain", "bsc",
                "hyperledger", "chainlink", "oracles", "dao", "decentralized autonomous organization", "yield farming", "liquidity mining",
                "layer 2", "rollups", "sidechains", "atomic swaps", "cross-chain", "interoperability",
                "rust blockchain", "go blockchain", "vyper", "web3.js", "ethers.js", "wallet integration"
            ],
            
            "Embedded Systems": [
                "embedded systems", "embedded programming", "microcontroller", "mcu", "firmware", "c programming", "embedded c", "assembly",
                "real-time systems", "rtos", "freertos", "arduino", "raspberry pi", "arm", "cortex", "pic", "avr",
                "hardware programming", "sensor integration", "iot devices", "internet of things",
                "low-level programming", "device drivers", "bootloader", "embedded linux", "yocto",
                "fpga", "verilog", "vhdl", "pcb design", "circuit design", "schematic design",
                "i2c", "spi", "uart", "can bus", "modbus", "gpio", "adc", "pwm", "timers", "interrupts",
                "power management", "battery optimization", "wireless communication", "bluetooth", "wifi", "lora"
            ],
            
            "System Architecture": [
                "system architecture", "solution architect", "software architect", "enterprise architecture", "microservices",
                "distributed systems", "scalability", "high availability", "fault tolerance", "reliability",
                "system design", "architecture patterns", "design patterns", "architectural design", "load balancing",
                "caching strategies", "database sharding", "event-driven architecture", "message queues",
                "api design", "service mesh", "containerization", "orchestration", "cloud architecture",
                "monolithic architecture", "modular architecture", "layered architecture", "hexagonal architecture",
                "clean architecture", "domain-driven design", "ddd", "cqrs", "event sourcing",
                "performance optimization", "capacity planning", "disaster recovery", "backup strategies"
            ],
            
            "Database Management": [
                "database administrator", "dba", "database management", "database design", "sql optimization", "query optimization",
                "database performance", "backup and recovery", "replication", "clustering", "sharding",
                "data modeling", "er modeling", "normalization", "denormalization", "indexing", "stored procedures", "triggers",
                "database security", "mysql", "postgresql", "postgres", "oracle", "sql server", "mongodb", "cassandra",
                "redis", "elasticsearch", "data warehouse", "etl", "extract transform load", "olap", "oltp",
                "mariadb", "sqlite", "neo4j", "graph database", "time series database", "influxdb",
                "database migration", "schema design", "database tuning", "connection pooling", "transaction management"
            ],
            
            "Networking": [
                "network engineer", "network administration", "networking", "cisco", "juniper", "routing", "switching",
                "tcp/ip", "dns", "dhcp", "vpn", "firewall", "network security", "network protocols",
                "network monitoring", "network troubleshooting", "wan", "lan", "vlan", "subnet", "subnetting",
                "bgp", "ospf", "eigrp", "rip", "mpls", "sd-wan", "network automation", "ccna", "ccnp", "ccie",
                "packet analysis", "wireshark", "network design", "bandwidth management", "qos", "quality of service",
                "load balancing", "network optimization", "wireless networking", "wifi", "802.11", "ethernet"
            ],
            
            "Site Reliability Engineering": [
                "sre", "site reliability", "site reliability engineering", "system reliability", "incident management",
                "post-mortem", "postmortem", "error budgets", "sli", "service level indicator", "slo", "service level objective", "monitoring", "alerting",
                "capacity planning", "performance optimization", "chaos engineering", "fault injection",
                "disaster recovery", "high availability", "fault tolerance", "observability", "reliability",
                "on-call", "incident response", "root cause analysis", "mttr", "mean time to recovery",
                "automation", "toil reduction", "reliability engineering", "production support"
            ],
            
            "Product Management": [
                "product manager", "product management", "product strategy", "product roadmap", "roadmap",
                "user stories", "requirements gathering", "stakeholder management", "product owner", "agile", "scrum",
                "scrum", "kanban", "product analytics", "metrics", "kpi", "a/b testing", "user research",
                "market research", "competitive analysis", "go-to-market", "gtm", "product launch", "product marketing",
                "feature prioritization", "backlog management", "product metrics", "user feedback",
                "product vision", "product discovery", "customer development", "lean startup", "mvp", "minimum viable product",
                "product-market fit", "user experience", "customer journey", "retention", "churn", "growth"
            ],
            
            "Project Management": [
                "project manager", "project management", "pmp", "project management professional", "agile", "scrum master",
                "kanban", "waterfall", "risk management", "resource planning", "timeline", "gantt chart",
                "milestone", "deliverables", "stakeholder communication", "budget management", "cost management",
                "team coordination", "project planning", "project execution", "project closure", "project monitoring",
                "change management", "quality assurance", "jira", "confluence", "ms project", "microsoft project",
                "prince2", "lean", "six sigma", "critical path", "work breakdown structure", "wbs"
            ],
            
            "Business Analysis": [
                "business analyst", "business analysis", "requirements analysis", "business requirements", "process improvement", "workflow",
                "business process", "stakeholder analysis", "gap analysis", "use cases", "user stories",
                "functional requirements", "non-functional requirements", "documentation", "business documentation",
                "process mapping", "business rules", "acceptance criteria", "user acceptance testing",
                "change management", "business intelligence", "data analysis", "reporting", "dashboard creation",
                "process optimization", "lean", "six sigma", "continuous improvement", "as-is", "to-be",
                "business case", "cost-benefit analysis", "roi", "return on investment", "feasibility study"
            ],
            
            "Technical Writing": [
                "technical writer", "technical writing", "documentation", "api documentation", "user manuals", "user guides",
                "technical communication", "content strategy", "information architecture", "technical documentation",
                "style guide", "editing", "proofreading", "copy editing", "markdown", "confluence", "notion",
                "gitbook", "sphinx", "doxygen", "technical blogging", "knowledge base", "help documentation",
                "instruction manuals", "process documentation", "software documentation", "developer documentation",
                "release notes", "change logs", "troubleshooting guides", "faq", "tutorial writing"
            ],
            
            "Digital Marketing": [
                "digital marketing", "seo", "search engine optimization", "sem", "search engine marketing", "social media marketing", "smm", "content marketing",
                "email marketing", "ppc", "pay per click", "google ads", "facebook ads", "instagram ads", "linkedin ads", "analytics",
                "conversion optimization", "cro", "marketing automation", "lead generation", "demand generation",
                "brand management", "influencer marketing", "affiliate marketing", "growth hacking", "growth marketing",
                "google analytics", "facebook pixel", "marketing funnel", "customer acquisition", "marketing attribution",
                "remarketing", "retargeting", "marketing campaigns", "digital advertising", "online marketing"
            ],
            
            "E-commerce": [
                "e-commerce", "ecommerce", "online retail", "online store", "shopify", "magento", "woocommerce", "bigcommerce",
                "payment gateway", "payment processing", "inventory management", "order management", "shipping", "fulfillment",
                "customer service", "marketplace", "amazon", "ebay", "etsy", "dropshipping", "conversion rate optimization",
                "product catalog", "shopping cart", "checkout optimization", "amazon fba", "fulfillment by amazon",
                "product photography", "product listings", "seo for ecommerce", "email marketing", "abandoned cart",
                "customer retention", "loyalty programs", "subscription commerce", "multi-channel selling"
            ],
            
            "Fintech": [
                "fintech", "financial technology", "payment processing", "digital payments", "banking software", "neobank",
                "trading systems", "algorithmic trading", "risk management", "compliance", "regulatory", "kyc", "know your customer",
                "aml", "anti-money laundering", "blockchain finance", "cryptocurrency", "digital wallet", "mobile payments",
                "robo-advisor", "wealth management", "insurtech", "regtech", "lending platform", "peer-to-peer lending",
                "credit scoring", "fraud detection", "financial analytics", "open banking", "api banking",
                "financial modeling", "quantitative finance", "high-frequency trading", "financial reporting"
            ],
            
            "Healthcare Tech": [
                "healthcare technology", "healthtech", "health tech", "medical software", "ehr", "electronic health records", "emr", "electronic medical records",
                "telemedicine", "telehealth", "medical devices", "hipaa", "healthcare analytics", "health informatics",
                "clinical trials", "medical imaging", "radiology", "bioinformatics", "genomics", "digital health",
                "patient management", "healthcare compliance", "medical ai", "clinical decision support",
                "mhealth", "mobile health", "wearables", "remote patient monitoring", "health data",
                "medical billing", "revenue cycle management", "laboratory information systems", "pharmacy systems"
            ],
            
            "EdTech": [
                "edtech", "educational technology", "e-learning", "online learning", "lms", "learning management system", "learning management",
                "online education", "educational software", "student information system", "sis", "learning platform",
                "assessment tools", "educational analytics", "learning analytics", "adaptive learning", "personalized learning", "gamification",
                "virtual classroom", "educational content", "curriculum development", "instructional design",
                "mooc", "massive open online course", "microlearning", "blended learning", "distance learning",
                "educational apps", "learning apps", "student engagement", "educational games", "simulation-based learning"
            ],
            
            "IoT Development": [
                "iot", "internet of things", "connected devices", "smart devices", "sensor networks", "sensor data",
                "edge computing", "fog computing", "mqtt", "coap", "zigbee", "bluetooth", "ble", "wifi", "lora", "lorawan",
                "embedded systems", "device management", "iot platform", "industrial iot", "iiot", "industry 4.0",
                "smart home", "home automation", "smart city", "smart grid", "wearables", "asset tracking", "predictive maintenance",
                "remote monitoring", "telemetry", "m2m", "machine to machine", "wireless communication", "sensor fusion",
                "real-time data", "time series data", "device connectivity", "iot security", "firmware over-the-air"
            ],
            
            "AR/VR Development": [
                "ar", "vr", "augmented reality", "virtual reality", "mixed reality", "mr", "xr", "extended reality",
                "unity 3d", "unreal engine", "oculus", "meta quest", "hololens", "magic leap", "arkit", "arcore",
                "3d modeling", "spatial computing", "immersive experience", "360 video", "stereoscopic",
                "haptic feedback", "motion tracking", "hand tracking", "eye tracking", "computer vision", "3d graphics", "webxr",
                "volumetric capture", "photogrammetry", "3d scanning", "virtual environments", "immersive storytelling",
                "vr training", "ar visualization", "spatial anchors", "occlusion", "slam", "simultaneous localization and mapping"
            ],
            
            "Technical Sales": [
                "technical sales", "sales engineer", "solution selling", "consultative selling", "pre-sales", "pre sales",
                "technical consulting", "customer success", "account management", "enterprise sales",
                "product demonstration", "technical presentation", "proposal writing", "rfp", "request for proposal",
                "client relationship", "revenue generation", "sales process", "crm", "customer relationship management",
                "lead qualification", "sales funnel", "pipeline management", "quota attainment", "b2b sales",
                "enterprise software sales", "saas sales", "solution architecture", "technical expertise"
            ],
            
            "Agile Coaching": [
                "agile coach", "agile coaching", "scrum master", "agile transformation", "team facilitation", "agile trainer",
                "retrospectives", "sprint planning", "daily standups", "scrum ceremonies", "agile ceremonies",
                "continuous improvement", "kaizen", "change management", "team dynamics", "servant leadership",
                "agile metrics", "velocity", "burn down", "coaching", "mentoring", "organizational change",
                "scaled agile", "safe", "less", "kanban coaching", "lean coaching", "agile mindset",
                "team building", "conflict resolution", "facilitation", "agile practices", "agile frameworks"
            ],
            
            "Software Engineering": [
                "software engineer", "software developer", "web developer", "developer", "programmer", "coder",
                "object oriented", "oop", "design patterns", "solid principles", "agile", "scrum", "git", "version control",
                "unit testing", "integration testing", "debugging", "code review", "system design", "software architecture",
                "tdd", "test driven development", "bdd", "behavior driven development", "pair programming", "refactoring", "uml", "dev environment", "ide",
                "algorithms", "data structures", "clean code", "software craftsmanship", "continuous integration",
                "software development", "full-stack", "backend", "frontend", "api development", "database design",
                "performance optimization", "scalability", "maintainability", "code quality", "technical debt"
            ]
        }

        # Explicit overrides - check these first before any other logic
        explicit_overrides = {
            "full stack developer": "Full Stack Development",
            "fullstack developer": "Full Stack Development", 
            "full-stack developer": "Full Stack Development",
            "mobile developer": "Mobile Development",
            "android developer": "Mobile Development",
            "ios developer": "Mobile Development",
            "react native developer": "Mobile Development"
        }
        
        # Check explicit overrides first
        for override_term, override_domain in explicit_overrides.items():
            if override_term in title or override_term in desc:
                logger.info(f"Explicit override detected: {override_term} -> {override_domain}")
                return override_domain

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

        # Step 4: Check for strong keywords in title regardless of description length
        strong_keywords_mapping = {
            "data scientist": "Data Science",
            "data analyst": "Data Science", 
            "ml engineer": "AI/Machine Learning",
            "ai engineer": "AI/Machine Learning",
            "frontend developer": "Frontend Development",
            "frontend engineer": "Frontend Development",
            "backend developer": "Backend Development", 
            "backend engineer": "Backend Development",
            "cloud engineer": "Cloud Engineering",
            "devops engineer": "DevOps/Infrastructure",
            "security analyst": "Cybersecurity",
            "mobile developer": "Mobile Development",
            "android developer": "Mobile Development", 
            "ios developer": "Mobile Development",
            "game developer": "Game Development",
            "blockchain developer": "Blockchain Development",
            "ui/ux designer": "UI/UX Design",
            "ux designer": "UI/UX Design",
            "ui designer": "UI/UX Design",
            "product manager": "Product Management",
            "project manager": "Project Management"
        }

        strong_keyword_detected = None
        for keyword, domain in strong_keywords_mapping.items():
            if keyword in title:
                strong_keyword_detected = domain
                break

        # Step 5: Filter short/noisy descriptions with improved handling
        desc_word_count = len(desc.split())
        if desc_word_count < 8:
            # If we have a strong keyword in title, use that regardless of short description
            if strong_keyword_detected:
                logger.info(f"Strong keyword detected in title despite short description: {strong_keyword_detected}")
                return strong_keyword_detected
            
            # Otherwise, reduce scores from description but keep title scores
            for domain in domain_scores:
                desc_hits = sum(1 for kw in keywords[domain] if kw in desc)
                domain_scores[domain] = max(0, domain_scores[domain] - (desc_hits * WEIGHTS[domain] * 0.5))

        # Step 6: Choose top domain with improved confidence threshold
        if domain_scores:
            top_domain = max(domain_scores, key=domain_scores.get)
            top_score = domain_scores[top_domain]
            
            # Apply new confidence threshold logic
            if top_score >= 8:
                logger.info(f"Domain detected: {top_domain} with score: {top_score}")
                return top_domain
            elif desc_word_count < 8 and not strong_keyword_detected:
                logger.info(f"Short description ({desc_word_count} words) and no strong keywords, returning General")
                return "General"
            else:
                logger.info(f"Low confidence detection ({top_score} < 8), returning Uncategorized")
                return "Uncategorized"

        # Final fallback
        if desc_word_count < 8:
            logger.info("No domain detected with short description, returning General")
            return "General"
        else:
            logger.info("No domain detected, returning Uncategorized")
            return "Uncategorized"

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
