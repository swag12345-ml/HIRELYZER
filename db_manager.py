import sqlite3
import pandas as pd
from datetime import datetime
import pytz
from collections import defaultdict

# 🔌 Connect to the SQLite database
conn = sqlite3.connect("resume_data.db", check_same_thread=False)
cursor = conn.cursor()

# 🗂️ Create the candidates table
cursor.execute("""
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_name TEXT,
    candidate_name TEXT,
    ats_score INTEGER,
    edu_score INTEGER,
    exp_score INTEGER,
    skills_score INTEGER,
    lang_score INTEGER,
    keyword_score INTEGER,
    bias_score REAL,
    domain TEXT,
    timestamp DATETIME
)
""")
conn.commit()

# ✅ Enhanced Domain Detection with 25+ Professional Domains
def detect_domain_from_title_and_description(job_title, job_description):
    title = job_title.lower().strip()
    desc = job_description.lower().strip()

    # 🔄 Enhanced normalization with more synonyms
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

    # 🎯 Enhanced weights for better domain differentiation
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

    # 🚀 Comprehensive keyword mapping for 30+ domains
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


# 🎯 Enhanced Domain Similarity with Comprehensive Mapping
def get_domain_similarity(resume_domain, job_domain):
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


# ✅ Enhanced insert function with better domain handling
def insert_candidate(data: tuple, job_title: str = "", job_description: str = ""):
    from datetime import datetime
    import pytz

    local_tz = pytz.timezone("Asia/Kolkata")
    local_time = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")

    # ✅ Detect domain from job title + description
    detected_domain = detect_domain_from_title_and_description(job_title, job_description)

    # ✅ Use only first 9 values and append domain
    normalized_data = data[:9] + (detected_domain,)

    # ✅ Insert into database
    cursor.execute("""
        INSERT INTO candidates (
            resume_name, candidate_name, ats_score, edu_score, exp_score,
            skills_score, lang_score, keyword_score, bias_score, domain, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, normalized_data + (local_time,))
    conn.commit()


# 📊 Top domains by ATS
def get_top_domains_by_score(limit: int = 5) -> list:
    cursor.execute("""
        SELECT domain, ROUND(AVG(ats_score), 2) AS avg_score, COUNT(*) AS count
        FROM candidates
        GROUP BY domain
        ORDER BY avg_score DESC
        LIMIT ?
    """, (limit,))
    return cursor.fetchall()

# 📈 Resume count by date (for trend analysis)
def get_resume_count_by_day():
    query = """
    SELECT DATE(timestamp) AS day, COUNT(*) AS count
    FROM candidates
    GROUP BY DATE(timestamp)
    ORDER BY DATE(timestamp) DESC
    """
    return pd.read_sql_query(query, conn)

# 📊 Average ATS score by domain
def get_average_ats_by_domain():
    query = """
    SELECT domain, ROUND(AVG(ats_score), 2) AS avg_ats_score
    FROM candidates
    GROUP BY domain
    ORDER BY avg_ats_score DESC
    """
    return pd.read_sql_query(query, conn)

# 🥧 Resume distribution by domain
def get_domain_distribution():
    query = """
    SELECT domain, COUNT(*) as count
    FROM candidates
    GROUP BY domain
    """
    return pd.read_sql_query(query, conn)

# 📅 Filter candidates by date
def filter_candidates_by_date(start: str, end: str):
    query = """
    SELECT * FROM candidates
    WHERE DATE(timestamp) BETWEEN DATE(?) AND DATE(?)
    ORDER BY timestamp DESC
    """
    return pd.read_sql_query(query, conn, params=(start, end))

# 🗑️ Delete candidate by ID
def delete_candidate_by_id(candidate_id: int):
    cursor.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
    conn.commit()

# 📄 Get all candidates with optional bias and ATS filters
def get_all_candidates(bias_threshold: float = None, min_ats: int = None):
    query = "SELECT * FROM candidates WHERE 1=1"
    params = []

    if bias_threshold is not None:
        query += " AND bias_score >= ?"
        params.append(bias_threshold)

    if min_ats is not None:
        query += " AND ats_score >= ?"
        params.append(min_ats)

    query += " ORDER BY timestamp DESC"
    return pd.read_sql_query(query, conn, params=params)

# 📤 Export all candidate data to CSV
def export_to_csv(filepath: str = "candidates_export.csv"):
    df = pd.read_sql_query("SELECT * FROM candidates ORDER BY timestamp DESC", conn)
    df.to_csv(filepath, index=False)

# 🔎 Get a specific candidate by ID (for safe delete preview)
def get_candidate_by_id(candidate_id: int):
    query = "SELECT * FROM candidates WHERE id = ?"
    return pd.read_sql_query(query, conn, params=(candidate_id,))

# 🧠 Get bias score distribution (Fair vs Biased resumes)
def get_bias_distribution(threshold: float = 0.6):
    query = f"""
    SELECT 
        CASE WHEN bias_score >= {threshold} THEN 'Biased' ELSE 'Fair' END AS bias_category,
        COUNT(*) AS count
    FROM candidates
    GROUP BY bias_category
    """
    return pd.read_sql_query(query, conn)

# 📈 ATS score trend over time
def get_daily_ats_stats():
    query = """
    SELECT DATE(timestamp) AS date, ROUND(AVG(ats_score), 2) AS avg_ats
    FROM candidates
    GROUP BY DATE(timestamp)
    ORDER BY DATE(timestamp)
    """
    return pd.read_sql_query(query, conn)

# 🚩 Get all flagged candidates (bias_score > threshold)
def get_flagged_candidates(threshold: float = 0.6):
    query = """
    SELECT resume_name, candidate_name, ats_score, bias_score, domain, timestamp
    FROM candidates
    WHERE bias_score > ?
    ORDER BY bias_score DESC
    """
    return pd.read_sql_query(query, conn, params=(threshold,))

# 🎯 Enhanced domain analytics
def get_domain_performance_stats():
    """Get comprehensive domain performance statistics"""
    query = """
    SELECT 
        domain,
        COUNT(*) as total_candidates,
        ROUND(AVG(ats_score), 2) as avg_ats_score,
        ROUND(AVG(edu_score), 2) as avg_edu_score,
        ROUND(AVG(exp_score), 2) as avg_exp_score,
        ROUND(AVG(skills_score), 2) as avg_skills_score,
        MAX(ats_score) as max_ats_score,
        MIN(ats_score) as min_ats_score
    FROM candidates
    GROUP BY domain
    ORDER BY avg_ats_score DESC
    """
    return pd.read_sql_query(query, conn)

# 🔍 Domain similarity analysis
def analyze_domain_transitions():
    """Analyze how candidates transition between domains"""
    query = """
    SELECT 
        domain,
        COUNT(*) as frequency,
        ROUND(AVG(ats_score), 2) as avg_performance
    FROM candidates
    GROUP BY domain
    ORDER BY frequency DESC
    """
    return pd.read_sql_query(query, conn)
S
