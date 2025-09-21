"): 0.70,
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

