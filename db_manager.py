"""
Enhanced Database Manager for Resume Analysis System
Optimized for large-scale user structures with improved performance and reliability
Enhanced Full Stack Detection and Comprehensive Domain Coverage
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

# Enhanced Domain Detection Configuration Constants
MIN_TOTAL_HITS = 2  # Reduced for better sensitivity
STRONG_HIT_WEIGHT = 4  # Increased weight for strong keywords
WEAK_HIT_WEIGHT = 1
FALLBACK_THRESHOLD_SCORE = 25  # Reduced threshold
FULLSTACK_TITLE_BOOST = 60  # Increased boost for full-stack titles
FULLSTACK_MENTION_BOOST = 35  # Increased boost for mentions
FULLSTACK_HITS_THRESHOLD_BOOST = 30  # Increased boost for combined hits
FULLSTACK_COMBO_BOOST = 40  # New boost for frontend+backend combinations

# Enhanced Domain Weights - Prioritizing Full Stack
DOMAIN_WEIGHTS = {
    "Full Stack Development": 5,  # Highest priority
    "Data Science": 4,
    "AI/Machine Learning": 4,
    "System Architecture": 4,
    "Cybersecurity": 4,
    "UI/UX Design": 3,
    "Mobile Development": 3,
    "Frontend Development": 3,
    "Backend Development": 3,
    "Cloud Engineering": 3,
    "DevOps/Infrastructure": 3,
    "Quality Assurance": 3,
    "Game Development": 3,
    "Blockchain Development": 3,
    "Embedded Systems": 3,
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

    def _validate_candidate_data(self, data: Tuple) -> Tuple:
        """
        Validate data tuple length, types, and ranges. Return normalized data tuple.
        """
        # Validate data length
        if len(data) < 9:
            raise ValueError(f"Expected at least 9 data fields, got {len(data)}")

        # Use only first 9 values
        validated_data = list(data[:9])

        # Validate score ranges (positions 2-7: ats_score to keyword_score)
        for i in range(2, 8):
            score = validated_data[i]
            if not isinstance(score, (int, float)):
                raise ValueError(f"Score at position {i} must be numeric, got {type(score)}")
            if not (0 <= score <= 100):
                raise ValueError(f"Score at position {i} must be between 0 and 100, got {score}")

        # Validate bias score (position 8)
        bias_score = validated_data[8]
        if not isinstance(bias_score, (int, float)):
            raise ValueError(f"Bias score must be numeric, got {type(bias_score)}")
        if not (0.0 <= bias_score <= 1.0):
            raise ValueError(f"Bias score must be between 0.0 and 1.0, got {bias_score}")

        # Validate text fields (positions 0-1: resume_name, candidate_name)
        for i in range(0, 2):
            if not isinstance(validated_data[i], str) or not validated_data[i].strip():
                raise ValueError(f"Text field at position {i} must be a non-empty string")

        return tuple(validated_data)

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
        Enhanced Domain Detection with robust heuristic-based mechanism
        Significantly improved Full Stack Detection and comprehensive keyword coverage
        """
        title = job_title.lower().strip()
        desc = job_description.lower().strip()

        # Enhanced normalization with more synonyms
        replacements = {
            "cyber security": "cybersecurity",
            "ai engineer": "artificial intelligence",
            "ml engineer": "machine learning",
            "software developer": "software engineer",
            "web developer": "full stack developer",
            "frontend developer": "frontend",
            "backend developer": "backend",
            "fullstack developer": "full stack developer",
            "full-stack developer": "full stack developer",
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
            "software architect": "system architecture",
            "full stack engineer": "full stack developer"
        }
        
        for old, new in replacements.items():
            title = title.replace(old, new)
            desc = desc.replace(old, new)

        domain_scores = defaultdict(int)

        # Comprehensive keyword mapping with significantly expanded coverage
        domain_keywords = {
            "Data Science": {
                "strong": [
                    "data scientist", "data analyst", "data science", "pandas", "numpy", 
                    "matplotlib", "seaborn", "power bi", "tableau", "looker", "sql", 
                    "jupyter", "databricks", "spark", "hadoop", "r programming",
                    "statistical modeling", "time series", "forecasting", "predictive analytics",
                    "data mining", "big data", "analytics", "statistical analysis", "quantitative analysis"
                ],
                "weak": [
                    "eda", "data analysis", "statistics", "data visualization", "excel", 
                    "dashboards", "insights", "hypothesis testing", "a/b testing", 
                    "business intelligence", "data wrangling", "feature engineering", 
                    "data storytelling", "exploratory analysis", "kpi", "metrics",
                    "analytics engineer", "etl", "data pipeline", "data warehouse", 
                    "olap", "oltp", "dimensional modeling", "data governance", "data quality",
                    "data transformation", "data cleaning", "correlation analysis", "regression analysis",
                    "cohort analysis", "funnel analysis", "churn analysis", "segmentation"
                ]
            },
            
            "AI/Machine Learning": {
                "strong": [
                    "machine learning", "ml engineer", "deep learning", "neural network",
                    "scikit-learn", "tensorflow", "pytorch", "llm", "huggingface", 
                    "xgboost", "lightgbm", "bert", "gpt", "yolo", "transformer", 
                    "autoencoder", "mistral", "llama", "openai", "langchain", "artificial intelligence",
                    "computer vision", "natural language processing", "deep neural networks"
                ],
                "weak": [
                    "nlp", "ai engineer", "classification", "regression", "clustering",
                    "reinforcement learning", "transfer learning", "model training", 
                    "ai models", "fine-tuning", "zero-shot", "one-shot", "vector embeddings", 
                    "prompt engineering", "mlops", "model deployment", "feature store", 
                    "model monitoring", "hyperparameter tuning", "ensemble methods", 
                    "gradient boosting", "random forest", "svm", "pca", "dimensionality reduction",
                    "supervised learning", "unsupervised learning", "semi-supervised learning",
                    "feature selection", "cross validation", "overfitting", "underfitting", "bias-variance tradeoff",
                    "recommendation systems", "anomaly detection", "sentiment analysis", "image recognition"
                ]
            },
            
            "UI/UX Design": {
                "strong": [
                    "ui", "ux", "figma", "designer", "user interface", "user experience",
                    "adobe xd", "sketch", "wireframe", "prototyping", "interaction design",
                    "visual design", "user research", "design systems"
                ],
                "weak": [
                    "usability", "accessibility", "human-centered design", "affinity diagram", 
                    "journey mapping", "heuristic evaluation", "persona", "responsive design", 
                    "mobile-first", "ux audit", "design tokens", "design thinking",
                    "information architecture", "card sorting", "tree testing", 
                    "user testing", "a/b testing design", "design sprint", "atomic design", 
                    "material design", "design ops", "brand design", "color theory", "typography",
                    "grid systems", "design patterns", "user flow", "customer journey", "empathy mapping",
                    "usability testing", "hci", "human computer interaction", "interface design"
                ]
            },
            
            "Mobile Development": {
                "strong": [
                    "android", "ios", "flutter", "kotlin", "swift", "mobile app", 
                    "react native", "mobile application", "xcode", "android studio",
                    "mobile developer", "app development", "mobile engineer"
                ],
                "weak": [
                    "play store", "app store", "firebase", "mobile sdk", "cross-platform", 
                    "native mobile", "push notifications", "in-app purchases", "mobile ui", 
                    "mobile ux", "apk", "ipa", "expo", "capacitor", "cordova", "xamarin", 
                    "ionic", "phonegap", "mobile testing", "app optimization", 
                    "mobile security", "offline functionality", "mobile analytics", 
                    "app monetization", "mobile performance", "gesture recognition", "touch interfaces",
                    "mobile frameworks", "responsive mobile", "progressive web app", "mobile first"
                ]
            },
            
            "Frontend Development": {
                "strong": [
                    "frontend", "html", "css", "javascript", "react", "angular", "vue",
                    "typescript", "next.js", "webpack", "bootstrap", "tailwind",
                    "frontend developer", "front-end", "client-side", "web development"
                ],
                "weak": [
                    "sass", "scss", "less", "es6", "es2015", "responsive design", "web accessibility", "dom", "jquery", 
                    "redux", "vite", "zustand", "framer motion", "storybook", "eslint", 
                    "vitepress", "pwa", "single page application", "spa", "csr", "ssr", "hydration", 
                    "component-based ui", "web components", "micro frontends", "bundler", 
                    "transpiler", "polyfill", "css grid", "flexbox", "css animations", 
                    "web performance", "lighthouse", "core web vitals", "module federation", "build tools",
                    "css preprocessors", "component library", "design tokens", "css-in-js", "styled components",
                    "browser compatibility", "cross-browser", "performance optimization", "lazy loading"
                ]
            },
            
            "Backend Development": {
                "strong": [
                    "backend", "node.js", "django", "flask", "express", "api development",
                    "java", "spring boot", "asp.net", "laravel", "go", "fastapi", "nest.js",
                    "backend developer", "back-end", "server-side", "api", "rest api"
                ],
                "weak": [
                    "sql", "nosql", "mysql", "postgresql", "mongodb", 
                    "graphql", "authentication", "authorization", "mvc", "orm",
                    "business logic", "database schema", "microservices", 
                    "websockets", "rabbitmq", "message broker", "cron jobs", "redis", 
                    "elasticsearch", "kafka", "grpc", "soap", "middleware", "caching",
                    "load balancing", "rate limiting", "api gateway", "serverless", 
                    "lambda functions", "database design", "data modeling", "server architecture",
                    "session management", "jwt", "oauth", "security", "scalability", "performance tuning"
                ]
            },
            
            "Full Stack Development": {
                "strong": [
                    "full stack", "fullstack", "full-stack", "mern", "mean", "mevn", "lamp", "jamstack",
                    "frontend and backend", "end-to-end development", "full stack developer",
                    "full stack engineer", "web application", "complete web development",
                    "frontend backend", "client server", "full web stack", "comprehensive web development"
                ],
                "weak": [
                    "api integration", "rest api", "graphql", "react + node", "vue + express",
                    "angular + spring", "react.js + express", "monolith", "microservices", 
                    "serverless architecture", "integrated app", "cross-functional development", 
                    "component-based architecture", "database design", "middleware", "mvc", "mvvm", 
                    "authentication", "authorization", "session management", "cloud deployment", 
                    "responsive ui", "performance tuning", "state management", "redux", 
                    "context api", "axios", "fetch api", "isomorphic", "universal rendering", 
                    "headless cms", "api-first development", "end to end web application", 
                    "complete web stack", "software architect", "web stack", "html css javascript",
                    "client-side server-side", "database integration", "ui backend integration",
                    "web services", "spa backend", "ssr", "deployment", "devops", "version control",
                    "agile development", "scrum", "project management", "technical leadership",
                    "code review", "testing", "debugging", "optimization", "security implementation",
                    "database administration", "server configuration", "api design", "user interface design"
                ]
            },
            
            "Cybersecurity": {
                "strong": [
                    "cybersecurity", "security analyst", "penetration testing", 
                    "ethical hacking", "owasp", "kali linux", "burp suite", "nmap", 
                    "wireshark", "information security", "cyber security", "security engineer"
                ],
                "weak": [
                    "vulnerability", "threat analysis", "infosec", "red team", "blue team",
                    "incident response", "firewall", "ids", "ips", "malware", "encryption",
                    "cyber threat", "security operations", "siem", "zero-day", "cyber attack",
                    "cve", "forensics", "security audit", "compliance", "ransomware", 
                    "threat hunting", "security architecture", "identity management", "pki", 
                    "security governance", "risk assessment", "vulnerability management", "soc",
                    "security policies", "data protection", "privacy", "gdpr", "hipaa", "pci compliance"
                ]
            },
            
            "Cloud Engineering": {
                "strong": [
                    "cloud", "aws", "azure", "gcp", "cloud engineer", "cloud computing",
                    "s3", "ec2", "terraform", "cloudwatch", "cloudtrail", "iam", "rds",
                    "cloud architect", "cloud infrastructure"
                ],
                "weak": [
                    "cloud security", "cloud formation", "load balancer", "auto scaling", 
                    "cloud storage", "cloud native", "cloud migration", "eks", "aks", "elb", 
                    "lambda", "azure functions", "cloud functions", "serverless", "containers", 
                    "cloud architecture", "multi-cloud", "hybrid cloud", "cloud cost optimization",
                    "docker", "kubernetes", "microservices", "saas", "paas", "iaas", "cdn",
                    "cloud monitoring", "disaster recovery", "backup solutions", "cloud networking"
                ]
            },
            
            "DevOps/Infrastructure": {
                "strong": [
                    "devops", "docker", "kubernetes", "ci/cd", "jenkins", "ansible",
                    "terraform", "prometheus", "grafana", "argocd", "helm", "devops engineer",
                    "infrastructure engineer", "platform engineer"
                ],
                "weak": [
                    "infrastructure as code", "monitoring", "deployment", "automation", 
                    "pipeline", "build and release", "scripting", "bash", "shell script", 
                    "site reliability", "sre", "fluxcd", "aws cli", "linux administration", 
                    "log aggregation", "observability", "splunk", "gitlab ci", 
                    "github actions", "azure devops", "puppet", "chef", "vagrant",
                    "infrastructure monitoring", "alerting", "incident management", 
                    "chaos engineering", "configuration management", "container orchestration",
                    "service mesh", "istio", "consul", "vault", "packer", "nomad"
                ]
            },
            
            "Quality Assurance": {
                "strong": [
                    "qa", "quality assurance", "testing", "test automation", "selenium", 
                    "cypress", "jest", "mocha", "junit", "testng", "postman", "jmeter", 
                    "appium", "qa engineer", "test engineer", "sdet"
                ],
                "weak": [
                    "test cases", "test planning", "bug tracking", "regression testing", 
                    "performance testing", "load testing", "stress testing", "api testing", 
                    "ui testing", "unit testing", "integration testing", "system testing", 
                    "acceptance testing", "test driven development", "tdd",
                    "behavior driven development", "bdd", "cucumber", "test management", 
                    "defect management", "test strategy", "qa processes", "manual testing",
                    "automated testing", "functional testing", "non-functional testing",
                    "usability testing", "security testing", "compatibility testing"
                ]
            },
            
            "Game Development": {
                "strong": [
                    "game development", "unity", "unreal engine", "c#", "c++", 
                    "game design", "game programming", "game developer", "game engineer"
                ],
                "weak": [
                    "3d modeling", "animation", "shader programming", "physics engine",
                    "game mechanics", "level design", "game testing", "multiplayer", 
                    "networking", "mobile games", "console games", "pc games", "vr games", 
                    "ar games", "game optimization", "performance profiling", 
                    "game analytics", "monetization", "gameplay", "ui/ux for games",
                    "game assets", "texture mapping", "lighting", "rendering", "audio programming"
                ]
            },
            
            "Blockchain Development": {
                "strong": [
                    "blockchain", "cryptocurrency", "smart contracts", "solidity", 
                    "ethereum", "bitcoin", "web3", "dapp", "blockchain developer",
                    "crypto developer", "defi developer"
                ],
                "weak": [
                    "defi", "nft", "consensus algorithms", "cryptography", 
                    "distributed ledger", "mining", "staking", "tokenomics", "metamask", 
                    "truffle", "hardhat", "ipfs", "polygon", "binance smart chain",
                    "hyperledger", "chainlink", "oracles", "dao", "yield farming",
                    "decentralized applications", "peer-to-peer", "hash functions", "merkle trees"
                ]
            },
            
            "Embedded Systems": {
                "strong": [
                    "embedded systems", "microcontroller", "firmware", "c programming", 
                    "assembly", "arduino", "raspberry pi", "arm", "pic", "embedded engineer",
                    "embedded developer", "firmware engineer"
                ],
                "weak": [
                    "real-time systems", "rtos", "embedded c", "hardware programming", 
                    "sensor integration", "iot devices", "low-level programming", 
                    "device drivers", "bootloader", "embedded linux", "fpga", "verilog", 
                    "vhdl", "pcb design", "circuit design", "interrupt handling",
                    "memory management", "power optimization", "hardware abstraction layer"
                ]
            },
            
            "System Architecture": {
                "strong": [
                    "system architecture", "solution architect", "enterprise architecture", 
                    "microservices", "distributed systems", "system design", "software architect",
                    "technical architect", "architecture design"
                ],
                "weak": [
                    "scalability", "high availability", "fault tolerance", 
                    "architecture patterns", "design patterns", "load balancing",
                    "caching strategies", "database sharding", "event-driven architecture", 
                    "message queues", "api design", "service mesh", "containerization", 
                    "orchestration", "cloud architecture", "monolithic architecture",
                    "service oriented architecture", "event sourcing", "cqrs", "circuit breaker"
                ]
            },
            
            "Database Management": {
                "strong": [
                    "database administrator", "dba", "database design", "sql optimization",
                    "mysql", "postgresql", "oracle", "sql server", "mongodb", "database engineer"
                ],
                "weak": [
                    "database performance", "backup and recovery", "replication", 
                    "clustering", "data modeling", "normalization", "indexing", 
                    "stored procedures", "triggers", "database security", "cassandra", 
                    "redis", "elasticsearch", "data warehouse", "etl", "olap", "oltp",
                    "query optimization", "database tuning", "partitioning", "sharding",
                    "acid properties", "transactions", "concurrency control"
                ]
            },
            
            "Networking": {
                "strong": [
                    "network engineer", "network administration", "cisco", "routing", 
                    "switching", "tcp/ip", "network architect", "network specialist"
                ],
                "weak": [
                    "dns", "dhcp", "vpn", "firewall", "network security",
                    "network monitoring", "network troubleshooting", "wan", "lan", "vlan",
                    "bgp", "ospf", "mpls", "sd-wan", "network automation", 
                    "network protocols", "subnetting", "vlsm", "qos", "bandwidth management",
                    "network design", "topology", "redundancy", "failover"
                ]
            },
            
            "Site Reliability Engineering": {
                "strong": [
                    "sre", "site reliability", "system reliability", "incident management",
                    "site reliability engineer", "platform reliability"
                ],
                "weak": [
                    "post-mortem", "error budgets", "sli", "slo", "monitoring", "alerting",
                    "capacity planning", "performance optimization", "chaos engineering",
                    "disaster recovery", "high availability", "fault tolerance", 
                    "observability", "on-call", "escalation", "runbooks", "automation",
                    "service level objectives", "mean time to recovery", "uptime"
                ]
            },
            
            "Product Management": {
                "strong": [
                    "product manager", "product management", "product strategy", "roadmap",
                    "product owner", "product lead"
                ],
                "weak": [
                    "user stories", "requirements gathering", "stakeholder management", 
                    "agile", "scrum", "kanban", "product analytics", "a/b testing", 
                    "user research", "market research", "competitive analysis", 
                    "go-to-market", "product launch", "feature prioritization", 
                    "backlog management", "kpi", "metrics", "product vision", "customer feedback",
                    "product lifecycle", "market analysis", "pricing strategy", "product positioning"
                ]
            },
            
            "Project Management": {
                "strong": [
                    "project manager", "project management", "pmp", "scrum master",
                    "jira", "confluence", "ms project", "project lead", "program manager"
                ],
                "weak": [
                    "agile", "kanban", "waterfall", "risk management", "resource planning", 
                    "timeline", "milestone", "deliverables", "stakeholder communication", 
                    "budget management", "team coordination", "project planning", 
                    "project execution", "project closure", "change management", 
                    "quality assurance", "project scheduling", "gantt charts", "critical path",
                    "scope management", "procurement", "vendor management"
                ]
            },
            
            "Business Analysis": {
                "strong": [
                    "business analyst", "requirements analysis", "process improvement",
                    "system analyst", "ba", "business systems analyst"
                ],
                "weak": [
                    "workflow", "business process", "stakeholder analysis", "gap analysis", 
                    "use cases", "functional requirements", "non-functional requirements", 
                    "documentation", "process mapping", "business rules", 
                    "acceptance criteria", "user acceptance testing", "change management", 
                    "business intelligence", "data analysis", "reporting", "business modeling",
                    "requirements elicitation", "traceability", "impact analysis"
                ]
            },
            
            "Technical Writing": {
                "strong": [
                    "technical writer", "documentation", "api documentation", "user manuals",
                    "technical communication", "content developer"
                ],
                "weak": [
                    "content strategy", "information architecture", "style guide", 
                    "editing", "proofreading", "markdown", "confluence", "gitbook", 
                    "sphinx", "doxygen", "technical blogging", "knowledge base",
                    "help documentation", "user guides", "tutorials", "release notes",
                    "technical specifications", "process documentation"
                ]
            },
            
            "Digital Marketing": {
                "strong": [
                    "digital marketing", "seo", "sem", "social media marketing", 
                    "google ads", "facebook ads", "digital marketer", "marketing specialist"
                ],
                "weak": [
                    "content marketing", "email marketing", "ppc", "analytics",
                    "conversion optimization", "marketing automation", "lead generation",
                    "brand management", "influencer marketing", "affiliate marketing", 
                    "growth hacking", "marketing campaigns", "customer acquisition",
                    "retention marketing", "marketing metrics", "roi", "ctr", "cpc"
                ]
            },
            
            "E-commerce": {
                "strong": [
                    "e-commerce", "online retail", "shopify", "magento", "woocommerce",
                    "ecommerce developer", "online store", "digital commerce"
                ],
                "weak": [
                    "payment gateway", "inventory management", "order management", 
                    "shipping", "customer service", "marketplace", "dropshipping", 
                    "conversion rate optimization", "product catalog", "shopping cart", 
                    "checkout optimization", "amazon fba", "online payments",
                    "merchant services", "omnichannel", "customer experience", "sales funnel"
                ]
            },
            
            "Fintech": {
                "strong": [
                    "fintech", "financial technology", "payment processing", 
                    "banking software", "trading systems", "fintech developer"
                ],
                "weak": [
                    "risk management", "compliance", "regulatory", "kyc", "aml", 
                    "blockchain finance", "cryptocurrency", "robo-advisor", "insurtech",
                    "lending platform", "credit scoring", "fraud detection", 
                    "financial analytics", "algorithmic trading", "portfolio management",
                    "financial modeling", "derivatives", "capital markets"
                ]
            },
            
            "Healthcare Tech": {
                "strong": [
                    "healthcare technology", "healthtech", "medical software", "ehr", "emr",
                    "telemedicine", "health tech developer", "medical systems"
                ],
                "weak": [
                    "medical devices", "hipaa", "healthcare analytics", "clinical trials", 
                    "medical imaging", "bioinformatics", "health informatics", 
                    "patient management", "healthcare compliance", "medical ai", 
                    "digital health", "clinical decision support", "patient data",
                    "medical records", "pharmacy systems", "laboratory systems"
                ]
            },
            
            "EdTech": {
                "strong": [
                    "edtech", "educational technology", "e-learning", "lms", 
                    "learning management", "education software", "edtech developer"
                ],
                "weak": [
                    "online education", "educational software", "student information system",
                    "assessment tools", "educational analytics", "adaptive learning", 
                    "gamification", "virtual classroom", "educational content", 
                    "curriculum development", "learning platforms", "mooc",
                    "instructional design", "learning outcomes", "student engagement"
                ]
            },
            
            "IoT Development": {
                "strong": [
                    "iot", "internet of things", "connected devices", "mqtt", "coap",
                    "iot developer", "iot engineer", "smart devices"
                ],
                "weak": [
                    "sensor networks", "edge computing", "zigbee", "bluetooth", "wifi",
                    "embedded systems", "device management", "iot platform", 
                    "industrial iot", "smart home", "smart city", "wearables", 
                    "asset tracking", "predictive maintenance", "telemetry",
                    "device connectivity", "sensor data", "actuators", "gateway devices"
                ]
            },
            
            "AR/VR Development": {
                "strong": [
                    "ar", "vr", "augmented reality", "virtual reality", "mixed reality", 
                    "xr", "unity 3d", "unreal engine", "oculus", "hololens",
                    "ar developer", "vr developer", "xr developer"
                ],
                "weak": [
                    "arkit", "arcore", "3d modeling", "spatial computing", 
                    "immersive experience", "360 video", "haptic feedback", 
                    "motion tracking", "computer vision", "3d graphics",
                    "stereoscopic rendering", "head tracking", "gesture recognition",
                    "spatial mapping", "immersive interfaces", "vr applications"
                ]
            },
            
            "Technical Sales": {
                "strong": [
                    "technical sales", "sales engineer", "solution selling", "pre-sales",
                    "technical sales engineer", "sales consultant"
                ],
                "weak": [
                    "technical consulting", "customer success", "account management",
                    "product demonstration", "technical presentation", "proposal writing",
                    "client relationship", "revenue generation", "sales process", "crm",
                    "lead qualification", "technical expertise", "customer requirements",
                    "solution design", "competitive positioning", "contract negotiation"
                ]
            },
            
            "Agile Coaching": {
                "strong": [
                    "agile coach", "scrum master", "agile transformation", "agile consultant"
                ],
                "weak": [
                    "team facilitation", "retrospectives", "sprint planning", 
                    "daily standups", "agile ceremonies", "continuous improvement", 
                    "change management", "team dynamics", "agile metrics", "coaching", 
                    "mentoring", "organizational change", "scaled agile", "safe",
                    "lean principles", "kanban", "agile practices", "servant leadership"
                ]
            },
            
            "Software Engineering": {
                "strong": [
                    "software engineer", "web developer", "developer", "programmer",
                    "git", "version control", "software developer", "coding"
                ],
                "weak": [
                    "object oriented", "design patterns", "agile", "scrum", "unit testing", 
                    "integration testing", "debugging", "code review", "system design",
                    "tdd", "bdd", "pair programming", "refactoring", "uml", 
                    "dev environment", "ide", "algorithms", "data structures", 
                    "software architecture", "clean code", "solid principles",
                    "continuous integration", "continuous deployment", "documentation"
                ]
            }
        }

        # Step 1: Compute weighted keyword matches with enhanced logic
        for domain, keywords in domain_keywords.items():
            # Count strong and weak hits in title and description
            strong_title_hits = sum(1 for kw in keywords["strong"] if kw in title)
            strong_desc_hits = sum(1 for kw in keywords["strong"] if kw in desc)
            weak_title_hits = sum(1 for kw in keywords["weak"] if kw in title)
            weak_desc_hits = sum(1 for kw in keywords["weak"] if kw in desc)
            
            # Enhanced weight calculation (6x for title strong, 2x for desc strong, 3x for title weak, 1x for desc weak)
            total_strong_hits = (6 * strong_title_hits + 2 * strong_desc_hits)
            total_weak_hits = (3 * weak_title_hits + 1 * weak_desc_hits)
            
            # Apply minimum total hits threshold
            total_hits = total_strong_hits + total_weak_hits
            if total_hits < MIN_TOTAL_HITS:
                domain_scores[domain] = 0
            else:
                # Apply improved scoring logic with strong/weak weights
                domain_scores[domain] = (total_strong_hits * STRONG_HIT_WEIGHT + 
                                       total_weak_hits * WEAK_HIT_WEIGHT) * DOMAIN_WEIGHTS[domain]

        # Step 2: Enhanced Full Stack Detection with aggressive scoring
        frontend_keywords = domain_keywords["Frontend Development"]
        backend_keywords = domain_keywords["Backend Development"]
        
        frontend_strong_hits = sum(1 for kw in frontend_keywords["strong"] if kw in title or kw in desc)
        frontend_weak_hits = sum(1 for kw in frontend_keywords["weak"] if kw in title or kw in desc)
        backend_strong_hits = sum(1 for kw in backend_keywords["strong"] if kw in title or kw in desc)
        backend_weak_hits = sum(1 for kw in backend_keywords["weak"] if kw in title or kw in desc)
        
        # Check for explicit full-stack mentions
        fullstack_explicit = any(term in title for term in [
            "full stack", "fullstack", "full-stack", "mern", "mean", "mevn", "lamp"
        ])
        fullstack_mentioned = any(term in title or term in desc for term in [
            "full stack", "fullstack", "full-stack", "frontend and backend", "end-to-end development"
        ])

        # Aggressive Full Stack Detection Logic
        if fullstack_explicit:
            domain_scores["Full Stack Development"] += FULLSTACK_TITLE_BOOST
        elif fullstack_mentioned:
            domain_scores["Full Stack Development"] += FULLSTACK_MENTION_BOOST

        # Boost for combined frontend+backend skills
        total_frontend_hits = frontend_strong_hits + frontend_weak_hits
        total_backend_hits = backend_strong_hits + backend_weak_hits
        
        if total_frontend_hits >= 3 and total_backend_hits >= 3:
            domain_scores["Full Stack Development"] += FULLSTACK_COMBO_BOOST
        elif total_frontend_hits >= 2 and total_backend_hits >= 2:
            domain_scores["Full Stack Development"] += FULLSTACK_HITS_THRESHOLD_BOOST

        # Additional full-stack indicators
        web_dev_indicators = sum(1 for term in [
            "web development", "web application", "web app", "website", "web stack",
            "javascript", "html", "css", "react", "node", "express", "database"
        ] if term in title or term in desc)
        
        if web_dev_indicators >= 4:
            domain_scores["Full Stack Development"] += 25

        # Check for technology stack combinations that indicate full-stack
        tech_stacks = [
            ["react", "node"], ["angular", "express"], ["vue", "node"],
            ["javascript", "python"], ["typescript", "node"], ["react", "python"],
            ["html", "css", "javascript"], ["frontend", "backend"], ["client", "server"]
        ]
        
        for stack in tech_stacks:
            if all(tech in title + " " + desc for tech in stack):
                domain_scores["Full Stack Development"] += 20

        # Step 3: Domain-specific boosts with expanded coverage
        domain_boosts = {
            "AI/Machine Learning": ["ai", "ml", "machine learning", "artificial intelligence", "neural", "deep learning"],
            "Cybersecurity": ["security", "cyber", "infosec", "penetration", "ethical hacking"],
            "Cloud Engineering": ["cloud", "aws", "azure", "gcp", "serverless", "containerization"],
            "Mobile Development": ["mobile", "android", "ios", "app", "flutter", "react native"],
            "Game Development": ["game", "unity", "unreal", "gaming", "3d"],
            "Blockchain Development": ["blockchain", "crypto", "web3", "defi", "smart contract"],
            "IoT Development": ["iot", "embedded", "sensor", "arduino", "raspberry"],
            "AR/VR Development": ["ar", "vr", "augmented", "virtual reality", "mixed reality"],
            "DevOps/Infrastructure": ["devops", "docker", "kubernetes", "ci/cd", "terraform"],
            "Data Science": ["data science", "analytics", "big data", "tableau", "power bi"]
        }

        for domain, boost_terms in domain_boosts.items():
            title_matches = sum(1 for term in boost_terms if term in title)
            desc_matches = sum(1 for term in boost_terms if term in desc)
            if title_matches > 0:
                domain_scores[domain] += 12 * title_matches
            if desc_matches > 0:
                domain_scores[domain] += 4 * desc_matches

        # Step 4: Penalty for very short descriptions (but less aggressive)
        if len(desc.split()) < 5:
            for domain in list(domain_scores.keys()):
                if domain != "Full Stack Development":  # Protect full-stack from penalties
                    domain_scores[domain] = max(0, domain_scores[domain] * 0.8)

        # Step 5: Special handling for ambiguous cases - favor full-stack when both frontend and backend are present
        if (domain_scores.get("Frontend Development", 0) > 0 and 
            domain_scores.get("Backend Development", 0) > 0 and
            domain_scores.get("Full Stack Development", 0) > 0):
            # Boost full-stack significantly when both frontend and backend are detected
            combined_score = (domain_scores["Frontend Development"] + domain_scores["Backend Development"]) * 0.8
            domain_scores["Full Stack Development"] = max(domain_scores["Full Stack Development"], combined_score)

        # Step 6: Choose top domain with enhanced fallback logic
        if domain_scores:
            top_domain = max(domain_scores, key=domain_scores.get)
            top_score = domain_scores[top_domain]
            
            # Lower threshold for full-stack detection
            if top_domain == "Full Stack Development" and top_score >= 15:
                return top_domain
            elif top_score >= FALLBACK_THRESHOLD_SCORE:
                return top_domain

        # Enhanced fallback logic - check for web development indicators
        web_indicators = ["web", "html", "css", "javascript", "website", "browser", "http", "url"]
        web_score = sum(1 for indicator in web_indicators if indicator in title or indicator in desc)
        if web_score >= 2:
            return "Full Stack Development"

        # Final fallback to general domain
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

        # Perfect match
        if resume_domain == job_domain:
            return 1.0

        # Enhanced similarity mapping with higher scores for full-stack relationships
        similarity_map = {
            # Full Stack relationships - Higher scores
            ("full stack development", "frontend development"): 0.95,
            ("full stack development", "backend development"): 0.95,
            ("full stack development", "ui/ux design"): 0.80,
            ("full stack development", "mobile development"): 0.75,
            ("full stack development", "software engineering"): 0.90,
            ("full stack development", "web development"): 0.98,
            ("full stack development", "cloud engineering"): 0.70,
            ("full stack development", "devops/infrastructure"): 0.65,
            
            # Frontend relationships
            ("frontend development", "ui/ux design"): 0.90,
            ("frontend development", "mobile development"): 0.70,
            ("frontend development", "software engineering"): 0.75,
            ("frontend development", "backend development"): 0.65,
            ("frontend development", "full stack development"): 0.95,
            
            # Backend relationships
            ("backend development", "database management"): 0.80,
            ("backend development", "cloud engineering"): 0.75,
            ("backend development", "devops/infrastructure"): 0.70,
            ("backend development", "system architecture"): 0.85,
            ("backend development", "software engineering"): 0.80,
            ("backend development", "full stack development"): 0.95,
            
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
            ("mobile development", "full stack development"): 0.75,
            
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
            ("system architecture", "full stack development"): 0.75,
            
            # Networking relationships
            ("networking", "cybersecurity"): 0.80,
            ("networking", "devops/infrastructure"): 0.75,
            ("networking", "system architecture"): 0.70,
            
            # Industry-specific relationships
            ("fintech", "software engineering"): 0.70,
            ("fintech", "backend development"): 0.75,
            ("fintech", "cybersecurity"): 0.70,
            ("fintech", "full stack development"): 0.75,
            ("healthcare tech", "software engineering"): 0.70,
            ("edtech", "software engineering"): 0.70,
            ("e-commerce", "full stack development"): 0.85,
            ("e-commerce", "backend development"): 0.75,
            
            # Sales & Communication relationships
            ("technical sales", "product management"): 0.65,
            ("technical writing", "business analysis"): 0.60,
            ("digital marketing", "business analysis"): 0.55,
            
            # General software relationships - Enhanced for full-stack
            ("software engineering", "full stack development"): 0.90,
            ("software engineering", "frontend development"): 0.75,
            ("software engineering", "backend development"): 0.80,
            ("software engineering", "mobile development"): 0.70,
            ("software engineering", "game development"): 0.70,
            ("software engineering", "quality assurance"): 0.75,
        }

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

        # Same category bonus with higher scores
        categories = [tech_domains, data_domains, infrastructure_domains, management_domains, design_domains]
        for category in categories:
            if resume_domain in category and job_domain in category:
                if category == tech_domains:
                    return 0.60  # Higher for tech domains
                return 0.50  # Moderate similarity for same category
        
        # Cross-category relationships
        if ((resume_domain in tech_domains and job_domain in infrastructure_domains) or
            (resume_domain in infrastructure_domains and job_domain in tech_domains)):
            return 0.50  # Increased from 0.45
        
        if ((resume_domain in data_domains and job_domain in tech_domains) or
            (resume_domain in tech_domains and job_domain in data_domains)):
            return 0.45  # Increased from 0.40

        # Default low similarity for unrelated domains
        return 0.25

    def insert_candidate(self, data: Tuple, job_title: str = "", job_description: str = "") -> int:
        """
        Enhanced insert function with better domain handling and error checking
        Returns the ID of the inserted candidate
        """
        try:
            # Validate candidate data first
            validated_data = self._validate_candidate_data(data)
            
            local_tz = pytz.timezone("Asia/Kolkata")
            local_time = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")

            # Detect domain from job title + description
            detected_domain = self.detect_domain_from_title_and_description(job_title, job_description)

            # Append domain to validated data
            final_data = validated_data + (detected_domain,)

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO candidates (
                        resume_name, candidate_name, ats_score, edu_score, exp_score,
                        skills_score, lang_score, keyword_score, bias_score, domain, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, final_data + (local_time,))
                conn.commit()
                candidate_id = cursor.lastrowid
                logger.info(f"Inserted candidate with ID: {candidate_id}, Domain: {detected_domain}")
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
    print("Enhanced Database Manager initialized successfully!")
    stats = get_database_stats()
    print(f"Database Statistics: {stats}")
    
    # Test full-stack detection
    test_cases = [
        ("Full Stack Developer", "React, Node.js, MongoDB, Express.js, HTML, CSS, JavaScript"),
        ("Software Engineer", "Frontend React, Backend Node.js, Database MySQL, API development"),
        ("Web Developer", "HTML, CSS, JavaScript, Python, Django, PostgreSQL"),
        ("Frontend Developer", "React, Angular, Vue.js, HTML, CSS, JavaScript, Responsive design"),
        ("Backend Developer", "Node.js, Express, MongoDB, API development, Authentication")
    ]
    
    print("\nTesting Enhanced Full Stack Detection:")
    for title, desc in test_cases:
        detected = detect_domain_from_title_and_description(title, desc)
        print(f"Title: '{title}' | Description: '{desc[:50]}...' | Detected: '{detected}'")
