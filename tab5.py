if tab5:
	with tab5:
		import sqlite3
		import pandas as pd
		import matplotlib.pyplot as plt
		import numpy as np
		import streamlit as st
		from datetime import datetime, timedelta
		import plotly.express as px
		import plotly.graph_objects as go
		from plotly.subplots import make_subplots
		import time
		import glob, os

		# Import enhanced database manager functions
		from db_manager import (
			get_top_domains_by_score,
			get_resume_count_by_day,
			get_average_ats_by_domain,
			get_domain_distribution,
			get_bias_distribution,
			filter_candidates_by_date,
			delete_candidate_by_id,
			get_all_candidates,
			get_candidate_by_id,
			get_domain_performance_stats,
			get_daily_ats_stats,
			get_flagged_candidates,
			get_database_stats,
			analyze_domain_transitions,
			export_to_csv,
			cleanup_old_records,
			DatabaseManager
		)

		# Initialize enhanced database manager
		@st.cache_resource
		def get_db_manager():
			return DatabaseManager()

		db_manager = get_db_manager()

		def create_enhanced_pie_chart(df, values_col, labels_col, title):
			"""Create an enhanced pie chart with better styling"""
			fig = px.pie(
				df, 
				values=values_col, 
				names=labels_col,
				title=title,
				color_discrete_sequence=px.colors.qualitative.Set3
			)
			fig.update_traces(
				textposition='inside', 
				textinfo='percent+label',
				hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
			)
			fig.update_layout(
				showlegend=True,
				legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01),
				margin=dict(t=50, b=50, l=50, r=150)
			)
			return fig

		def create_enhanced_bar_chart(df, x_col, y_col, title, orientation='v'):
			"""Create enhanced bar chart with better interactivity"""
			if orientation == 'v':
				fig = px.bar(df, x=x_col, y=y_col, title=title, 
							color=y_col, color_continuous_scale='viridis')
				fig.update_xaxes(tickangle=45)
			else:
				fig = px.bar(df, x=y_col, y=x_col, title=title, orientation='h',
							color=y_col, color_continuous_scale='viridis')
			
			fig.update_traces(
				hovertemplate='<b>%{y if orientation == "v" else x}</b><br>Value: %{x if orientation == "v" else y}<extra></extra>'
			)
			fig.update_layout(showlegend=False, margin=dict(t=50, b=50, l=50, r=50))
			return fig

		def load_domain_distribution():
			"""Enhanced domain distribution loading with error handling"""
			try:
				df = get_domain_distribution()
				if not df.empty:
					df = df.sort_values(by="count", ascending=False).reset_index(drop=True)
					return df
			except Exception as e:
				st.error(f"Error loading domain distribution: {e}")
			return pd.DataFrame()

		# Enhanced Data Loading with Caching
		@st.cache_data(ttl=300)  # Cache for 5 minutes
		def load_all_candidates():
			try:
				return get_all_candidates()
			except Exception as e:
				st.error(f"Error loading candidates: {e}")
				return pd.DataFrame()

		# -------- Glassmorphism Styles with Shimmer --------
		st.markdown("""
		<style>
		.glass-box {
			background: rgba(10, 20, 40, 0.55);
			border-radius: 18px;
			padding: 2rem;
			backdrop-filter: blur(14px);
			border: 1px solid rgba(0, 200, 255, 0.35);
			box-shadow: 0 8px 32px rgba(0, 200, 255, 0.25);
			position: relative;
			overflow: hidden;
			text-align: center;
			margin-bottom: 2rem;
		}
		.glass-box::before {
			content: "";
			position: absolute;
			top: -50%;
			left: -50%;
			width: 200%;
			height: 200%;
			background: linear-gradient(
				120deg,
				rgba(255,255,255,0.15) 0%,
				rgba(255,255,255,0.05) 40%,
				transparent 60%
			);
			transform: rotate(25deg);
			animation: shimmer 6s infinite;
		}
		@keyframes shimmer {
			0% { top: -50%; left: -50%; }
			50% { top: 100%; left: 100%; }
			100% { top: -50%; left: -50%; }
		}
		.glass-box h1, .glass-box h2 {
			color: #4da6ff;
			text-shadow: 0 0 12px rgba(0,200,255,0.7);
			margin: 0 0 0.5rem 0;
			font-weight: 600;
		}
		.glass-box p {
			color: #cce6ff;
			margin: 0;
			font-size: 0.95rem;
		}

		/* Glassy input fields */
		.stTextInput > div > div > input {
			background: rgba(255, 255, 255, 0.08) !important;
			border: 1px solid rgba(0, 200, 255, 0.3) !important;
			border-radius: 12px !important;
			padding: 10px !important;
			color: #e6f7ff !important;
			font-weight: 500 !important;
			backdrop-filter: blur(10px) !important;
		}
		.stTextInput > div > div > input:focus {
			border: 1px solid rgba(0, 200, 255, 0.8) !important;
			box-shadow: 0 0 12px rgba(0, 200, 255, 0.6) !important;
			outline: none !important;
		}

		/* Glassy button */
		.stButton > button {
			background: rgba(0, 200, 255, 0.15);
			border: 1px solid rgba(0, 200, 255, 0.4);
			border-radius: 12px;
			color: #e6f7ff;
			padding: 0.6rem 1.2rem;
			font-weight: bold;
			backdrop-filter: blur(8px);
			transition: all 0.3s ease;
		}
		.stButton > button:hover {
			background: rgba(0, 200, 255, 0.3);
			box-shadow: 0 0 16px rgba(0, 200, 255, 0.7);
			transform: translateY(-2px);
		}
		</style>
		""", unsafe_allow_html=True)

		# ---------------- Enhanced Authentication System ----------------
		if "admin_logged_in" not in st.session_state:
			st.session_state.admin_logged_in = False

		if not st.session_state.admin_logged_in:
			st.markdown("""
			<div class="glass-box">
				<h2>🔐 Admin Authentication Required</h2>
				<p>Please enter your email and password to access the admin dashboard</p>
			</div>
			""", unsafe_allow_html=True)
			
			col1, col2, col3 = st.columns([1, 2, 1])
			with col2:
				email = st.text_input("📧 Enter Admin Email", placeholder="Enter email...")
				password = st.text_input("🔑 Enter Admin Password", type="password", placeholder="Enter password...")
				login_clicked = st.button("🚀 Login", use_container_width=True)

				if login_clicked:
					valid_email = "admin@example.com"
					valid_password = "Swagato@2002"

					if email == valid_email and password == valid_password:
						st.session_state.admin_logged_in = True
						st.success("✅ Authentication successful! Redirecting to dashboard...")
						st.rerun()
					else:
						msg_placeholder = st.empty()
						msg_placeholder.markdown("""
							<div style='
								background-color: #ff4d4d;
								color: white;
								padding: 10px 15px;
								border-radius: 10px;
								text-align: center;
								animation: slideDown 0.5s ease-in-out;
							'>❌ Invalid credentials. Please try again.</div>
							<style>
							@keyframes slideDown {
								0% {transform: translateY(-50px); opacity: 0;}
								100% {transform: translateY(0); opacity: 1;}
							}
							</style>
						""", unsafe_allow_html=True)
						time.sleep(3)
						msg_placeholder.empty()

			st.stop()

		# ---------------- Enhanced Header with Database Stats ----------------
		st.markdown("""
		<div class="glass-box">
			<h1>🛡️ Enhanced Admin Database Panel</h1>
			<p>Advanced Resume Analysis System Dashboard</p>
		</div>
		""", unsafe_allow_html=True)

		# Enhanced Control Panel
		col1, col2, col3, col4 = st.columns(4)
		with col1:
			if st.button("🔄 Refresh All Data", use_container_width=True):
				st.cache_data.clear()
				st.rerun()
		with col2:
			if st.button("📊 Database Stats", use_container_width=True):
				st.session_state.show_db_stats = True
		with col3:
			if st.button("🧹 Cleanup Old Records", use_container_width=True):
				st.session_state.show_cleanup = True
		with col4:
			if st.button("🚪 Secure Logout", use_container_width=True):
				st.session_state.admin_logged_in = False
				st.success("👋 Logged out successfully.")
				st.rerun()

		# Database Statistics Panel
		if st.session_state.get('show_db_stats', False):
			with st.expander("📈 Database Statistics", expanded=True):
				try:
					stats = get_database_stats()
					if stats:
						col1, col2, col3, col4 = st.columns(4)
						with col1:
							st.metric("Total Candidates", stats.get('total_candidates', 0))
						with col2:
							st.metric("Average ATS Score", f"{stats.get('avg_ats_score', 0):.2f}")
						with col3:
							st.metric("Unique Domains", stats.get('unique_domains', 0))
						with col4:
							st.metric("Database Size", f"{stats.get('database_size_mb', 0):.2f} MB")
						
						col5, col6 = st.columns(2)
						with col5:
							st.metric("Earliest Record", stats.get('earliest_date', 'N/A'))
						with col6:
							st.metric("Latest Record", stats.get('latest_date', 'N/A'))
				except Exception as e:
					st.error(f"Error loading database statistics: {e}")

		# Cleanup Panel
		if st.session_state.get('show_cleanup', False):
			with st.expander("🧹 Database Cleanup", expanded=True):
				days_to_keep = st.slider("Days to Keep", 30, 730, 365)
				if st.button("⚠️ Cleanup Old Records"):
					try:
						deleted_count = cleanup_old_records(days_to_keep)
						if deleted_count > 0:
							st.success(f"✅ Cleaned up {deleted_count} old records")
						else:
							st.info("ℹ️ No old records found to cleanup")
					except Exception as e:
						st.error(f"Error during cleanup: {e}")

		st.markdown("<hr style='border-top: 2px solid #bbb; margin: 2rem 0;'>", unsafe_allow_html=True)

		df = load_all_candidates()

		# Enhanced Search and Filter Section
		st.markdown("### 🔍 Advanced Search & Filters")
		
		col1, col2 = st.columns(2)
		with col1:
			search = st.text_input("🔍 Search by Candidate Name", placeholder="Enter candidate name...")
			if search:
				df = df[df["candidate_name"].str.contains(search, case=False, na=False)]
		
		with col2:
			domain_filter = st.selectbox("🏢 Filter by Domain", 
									options=["All Domains"] + list(df["domain"].unique()) if not df.empty else ["All Domains"])
			if domain_filter != "All Domains":
				df = df[df["domain"] == domain_filter]

		# Enhanced Date Filter
		st.markdown("#### 📅 Date Range Filter")
		col1, col2, col3 = st.columns(3)
		with col1:
			start_date = st.date_input("📅 Start Date", value=datetime.now() - timedelta(days=30))
		with col2:
			end_date = st.date_input("📅 End Date", value=datetime.now())
		with col3:
			if st.button("🎯 Apply Filters", use_container_width=True):
				try:
					df = filter_candidates_by_date(str(start_date), str(end_date))
					if domain_filter != "All Domains":
						df = df[df["domain"] == domain_filter]
					if search:
						df = df[df["candidate_name"].str.contains(search, case=False, na=False)]
					st.success(f"✅ Filters applied. Found {len(df)} candidates.")
				except Exception as e:
					st.error(f"Error applying filters: {e}")

		# Enhanced Candidates Display
		if df.empty:
			st.info("ℹ️ No candidate data available with current filters.")
		else:
			st.markdown(f"### 📋 Candidates Overview ({len(df)} records)")
			
			# Enhanced metrics
			col1, col2, col3, col4 = st.columns(4)
			with col1:
				st.metric("Total Candidates", len(df))
			with col2:
				st.metric("Avg ATS Score", f"{df['ats_score'].mean():.2f}")
			with col3:
				st.metric("Avg Bias Score", f"{df['bias_score'].mean():.3f}")
			with col4:
				st.metric("Unique Domains", df['domain'].nunique())

			# Enhanced data display with sorting
			sort_column = st.selectbox("📊 Sort by", 
								options=['timestamp', 'ats_score', 'bias_score', 'candidate_name', 'domain'])
			sort_order = st.radio("Sort Order", ["Descending", "Ascending"], horizontal=True)
			
			df_sorted = df.sort_values(by=sort_column, ascending=(sort_order == "Ascending"))
			
			# Display with enhanced formatting
			st.dataframe(
				df_sorted.style.format({
					'ats_score': '{:.0f}',
					'edu_score': '{:.0f}',
					'exp_score': '{:.0f}',
					'skills_score': '{:.0f}',
					'lang_score': '{:.0f}',
					'keyword_score': '{:.0f}',
					'bias_score': '{:.3f}'
				}),
				use_container_width=True,
				height=400
			)

			# Enhanced Export Options
			col1, col2 = st.columns(2)
			with col1:
				csv_data = df_sorted.to_csv(index=False)
				st.download_button(
					label="📥 Download Filtered Data (CSV)",
					data=csv_data,
					file_name=f"candidates_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
					mime="text/csv",
					use_container_width=True
				)
			with col2:
				if st.button("📤 Export All Data", use_container_width=True):
					try:
						filename = f"full_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
						if export_to_csv(filename):
							st.success(f"✅ Data exported to {filename}")
						else:
							st.error("❌ Export failed")
					except Exception as e:
						st.error(f"Export error: {e}")

			st.markdown("### 📂 Export Archive")
			export_files = sorted(glob.glob("full_export_*.csv"), reverse=True)

			if export_files:
				for file in export_files:
					with open(file, "rb") as f:
						st.download_button(
							label=f"⬇️ Download {os.path.basename(file)}",
							data=f,
							file_name=os.path.basename(file),
							mime="text/csv",
							use_container_width=True
						)
			else:
				st.info("📭 No export files found yet.")

			# Enhanced Delete Functionality
			with st.expander("🗑️ Delete Candidate", expanded=False):
				st.warning("⚠️ This action cannot be undone!")
				delete_id = st.number_input("Enter Candidate ID", min_value=1, step=1, key="delete_id")
				
				if delete_id in df["id"].values:
					candidate_info = get_candidate_by_id(delete_id)
					if not candidate_info.empty:
						st.info("📄 Candidate to be deleted:")
						st.dataframe(candidate_info, use_container_width=True)
						
						if st.button("❌ Confirm Delete", type="primary"):
							try:
								if delete_candidate_by_id(delete_id):
									st.success(f"✅ Candidate with ID {delete_id} deleted successfully.")
									st.cache_data.clear()
									st.rerun()
								else:
									st.error("❌ Failed to delete candidate.")
							except Exception as e:
								st.error(f"Delete error: {e}")
				elif delete_id > 0:
					st.error("❌ Candidate ID not found.")

		# Enhanced Analytics Section
		st.markdown("<hr style='border-top: 2px solid #bbb; margin: 2rem 0;'>", unsafe_allow_html=True)
		st.markdown("## 📊 Advanced Analytics Dashboard")

		# Enhanced Top Domains Analysis
		st.markdown("### 🏆 Top Performing Domains")
		
		try:
			top_domains = get_top_domains_by_score(limit=10)
			if top_domains:
				df_top = pd.DataFrame(top_domains, columns=["domain", "avg_ats", "count"])
				
				col1, col2 = st.columns([1, 2])
				with col1:
					sort_order = st.radio("📊 Sort by ATS", ["⬆️ High to Low", "⬇️ Low to High"], horizontal=True)
					limit = st.slider("Show Top N Domains", 1, len(df_top), value=min(8, len(df_top)))
				
				ascending = sort_order == "⬇️ Low to High"
				df_sorted = df_top.sort_values(by="avg_ats", ascending=ascending).head(limit)
				
				# Interactive chart
				fig = create_enhanced_bar_chart(df_sorted, "domain", "avg_ats", 
										"Average ATS Score by Domain", orientation='h')
				st.plotly_chart(fig, use_container_width=True)
				
				# Enhanced domain cards
				for i, row in df_sorted.iterrows():
					progress_value = row['avg_ats'] / 100
					st.markdown(f"""
					<div style="border: 2px solid #e1e5e9; border-radius: 15px; padding: 15px; margin-bottom: 15px; 
								background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);">
						<div style="display: flex; justify-content: space-between; align-items: center;">
							<h4 style="margin: 0; color: #495057;">📁 {row['domain']}</h4>
							<span style="background: #007bff; color: white; padding: 5px 10px; border-radius: 20px; font-size: 12px;">
								Rank #{i+1}
							</span>
						</div>
						<div style="margin: 10px 0;">
							<div style="background: #e9ecef; border-radius: 10px; height: 8px; overflow: hidden;">
								<div style="background: linear-gradient(90deg, #28a745, #20c997); height: 100%; 
									width: {progress_value*100}%; transition: width 0.3s ease;"></div>
							</div>
						</div>
						<div style="display: flex; justify-content: space-between; margin-top: 10px;">
							<span><b>🧠 Avg ATS:</b> <span style="color:#007acc; font-weight: bold;">{row['avg_ats']:.2f}</span></span>
							<span><b>📄 Resumes:</b> {row['count']}</span>
						</div>
					</div>
					""", unsafe_allow_html=True)
			else:
				st.info("ℹ️ No domain performance data available.")
		except Exception as e:
			st.error(f"Error loading top domains: {e}")

		# Enhanced Domain Distribution
		st.markdown("### 📊 Domain Distribution Analysis")

		try:
			df_domain_dist = load_domain_distribution()
			if not df_domain_dist.empty:
				col1, col2 = st.columns(2)
				with col1:
					chart_type = st.radio(
						"📊 Visualization Type:",
						["📈 Interactive Bar Chart", "🥧 Interactive Pie Chart"],
						horizontal=True
					)
				with col2:
					max_val = len(df_domain_dist)
					if max_val <= 5:
						show_top_n = max_val  # No slider, just show all available domains
					else:
						show_top_n = st.slider(
							"Show Top N Domains",
							min_value=5,
							max_value=max_val,
							value=min(10, max_val)
						)

				df_top_domains = df_domain_dist.head(show_top_n)

				if chart_type == "📈 Interactive Bar Chart":
					fig = create_enhanced_bar_chart(df_top_domains, "domain", "count", 
											"Resume Count by Domain")
					st.plotly_chart(fig, use_container_width=True)
				else:
					fig = create_enhanced_pie_chart(df_top_domains, "count", "domain", 
											"Domain Distribution")
					st.plotly_chart(fig, use_container_width=True)

				# Summary statistics
				with st.expander("📋 Domain Statistics Summary"):
					st.dataframe(
						df_domain_dist.style.format({'percentage': '{:.2f}%'}),
						use_container_width=True
					)
			else:
				st.info("ℹ️ No domain distribution data available.")
		except Exception as e:
			st.error(f"Error loading domain distribution: {e}")

		# Enhanced ATS Performance Analysis
		st.markdown("### 📈 ATS Performance Analysis")
		
		try:
			df_ats = get_average_ats_by_domain()
			if not df_ats.empty:
				col1, col2 = st.columns(2)
				with col1:
					chart_orientation = st.radio("Chart Style", ["Vertical", "Horizontal"], horizontal=True)
				with col2:
					color_scheme = st.selectbox("Color Scheme", 
										["plasma", "viridis", "inferno", "magma", "turbo"])
				
				orientation = 'v' if chart_orientation == "Vertical" else 'h'
				fig = px.bar(df_ats, 
							x="domain" if orientation == 'v' else "avg_ats_score",
							y="avg_ats_score" if orientation == 'v' else "domain",
							title="Average ATS Score by Domain",
							orientation=orientation,
							color="avg_ats_score",
							color_continuous_scale=color_scheme,
							text="avg_ats_score",
							template="plotly_dark")  # Use dark theme for better readability
				
				fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
				if orientation == 'v':
					fig.update_xaxes(tickangle=45)
				
				# Enhanced layout for better readability
				fig.update_layout(
					showlegend=False,
					plot_bgcolor='rgba(0,0,0,0.1)',
					paper_bgcolor='rgba(0,0,0,0.05)',
					font=dict(color='white', size=12),
					title=dict(font=dict(size=16, color='white')),
					xaxis=dict(
						gridcolor='rgba(255,255,255,0.2)',
						tickfont=dict(color='white')
					),
					yaxis=dict(
						gridcolor='rgba(255,255,255,0.2)',
						tickfont=dict(color='white')
					),
					margin=dict(t=60, b=80, l=80, r=50)
				)
				
				st.plotly_chart(fig, use_container_width=True)
			else:
				st.info("ℹ️ No ATS performance data available.")
		except Exception as e:
			st.error(f"Error loading ATS performance data: {e}")

		# Enhanced Timeline Analysis
		st.markdown("### 📈 Resume Upload Timeline & Trends")
		
		try:
			df_timeline = get_resume_count_by_day()
			df_daily_ats = get_daily_ats_stats(days_limit=90)
			
			if not df_timeline.empty:
				df_timeline = df_timeline.sort_values("day")
				df_timeline["7_day_avg"] = df_timeline["count"].rolling(window=7, min_periods=1).mean()
				df_timeline["30_day_avg"] = df_timeline["count"].rolling(window=30, min_periods=1).mean()
				
				# Create subplot with proper spacing and formatting
				fig = make_subplots(
					rows=2, cols=1,
					subplot_titles=('Daily Upload Count with Moving Averages', 'Daily Average ATS Score Trend'),
					vertical_spacing=0.25,
					specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
				)
				
				# Convert day column to datetime for proper spacing
				df_timeline['day'] = pd.to_datetime(df_timeline['day'])
				
				# Upload count plot
				fig.add_trace(
					go.Scatter(x=df_timeline["day"], y=df_timeline["count"], 
								mode='lines+markers', name='Daily Uploads',
								line=dict(color='#1f77b4', width=2),
								marker=dict(size=6)),
					row=1, col=1
				)
				
				fig.add_trace(
					go.Scatter(x=df_timeline["day"], y=df_timeline["7_day_avg"], 
								mode='lines', name='7-Day Average',
								line=dict(color='#ff7f0e', width=2, dash='dash')),
					row=1, col=1
				)
				
				fig.add_trace(
					go.Scatter(x=df_timeline["day"], y=df_timeline["30_day_avg"], 
								mode='lines', name='30-Day Average',
								line=dict(color='#2ca02c', width=2, dash='dot')),
					row=1, col=1
				)
				
				# ATS trend plot
				if not df_daily_ats.empty:
					df_daily_ats['date'] = pd.to_datetime(df_daily_ats['date'])
					fig.add_trace(
						go.Scatter(x=df_daily_ats["date"], y=df_daily_ats["avg_ats"], 
									mode='lines+markers', name='Daily Avg ATS',
									line=dict(color='#d62728', width=2),
									marker=dict(size=6)),
						row=2, col=1
					)
				
				# Update layout for better spacing and readability
				fig.update_layout(
					height=700, 
					showlegend=True,
					legend=dict(
						orientation="h",
						yanchor="bottom",
						y=1.02,
						xanchor="right",
						x=1
					),
					margin=dict(t=80, b=70, l=50, r=50)
				)
				
				# Update x-axes for proper date formatting and spacing
				fig.update_xaxes(title_text="Date", row=2, col=1)
				fig.update_xaxes(
					tickformat="%Y-%m-%d",
					tickangle=30,
					dtick="D1" if len(df_timeline) <= 30 else "D7",
					row=1, col=1
				)
				fig.update_xaxes(
					tickformat="%Y-%m-%d",
					tickangle=30,
					dtick="D1" if len(df_daily_ats) <= 30 else "D7",
					row=2, col=1
				)
				
				fig.update_yaxes(title_text="Upload Count", row=1, col=1)
				fig.update_yaxes(title_text="Average ATS Score", row=2, col=1)
				
				st.plotly_chart(fig, use_container_width=True)
				
				# Timeline statistics
				col1, col2, col3, col4 = st.columns(4)
				with col1:
					st.metric("Total Days", len(df_timeline))
				with col2:
					st.metric("Peak Daily Uploads", df_timeline["count"].max())
				with col3:
					st.metric("Avg Daily Uploads", f"{df_timeline['count'].mean():.1f}")
				with col4:
					if not df_daily_ats.empty:
						st.metric("Avg ATS Trend", f"{df_daily_ats['avg_ats'].mean():.2f}")
			else:
				st.info("ℹ️ No timeline data available.")
		except Exception as e:
			st.error(f"Error loading timeline data: {e}")

		# Enhanced Bias Analysis
		st.markdown("### 🧠 Advanced Bias Analysis")
		
		col1, col2 = st.columns(2)
		with col1:
			bias_threshold_pie = st.slider("Bias Detection Threshold", 
									min_value=0.0, max_value=1.0, value=0.6, step=0.05)
		with col2:
			analysis_type = st.radio("Analysis Type", ["Distribution", "Flagged Candidates"], horizontal=True)
		
		try:
			if analysis_type == "Distribution":
				df_bias = get_bias_distribution(threshold=bias_threshold_pie)
				if not df_bias.empty and "bias_category" in df_bias.columns:
					fig = create_enhanced_pie_chart(df_bias, "count", "bias_category", 
											f"Bias Distribution (Threshold: {bias_threshold_pie})")
					st.plotly_chart(fig, use_container_width=True)
					
					# Bias statistics
					col1, col2 = st.columns(2)
					with col1:
						total_candidates = df_bias["count"].sum()
						biased_count = df_bias[df_bias["bias_category"] == "Biased"]["count"].iloc[0] if len(df_bias[df_bias["bias_category"] == "Biased"]) > 0 else 0
						st.metric("Total Analyzed", total_candidates)
					with col2:
						bias_percentage = (biased_count / total_candidates * 100) if total_candidates > 0 else 0
						st.metric("Bias Percentage", f"{bias_percentage:.1f}%")
				else:
					st.info("📭 No bias distribution data available.")
			
			else:  # Flagged Candidates
				flagged_df = get_flagged_candidates(threshold=bias_threshold_pie)
				if not flagged_df.empty:
					st.markdown(f"**🚩 {len(flagged_df)} candidates flagged with bias score > {bias_threshold_pie}**")
					
					# Enhanced flagged candidates display
					display_df = flagged_df.copy()
					display_df = display_df.sort_values('bias_score', ascending=False)
					
					st.dataframe(
						display_df.style.format({'bias_score': '{:.3f}', 'ats_score': '{:.0f}'}),
						use_container_width=True,
						height=300
					)
					
					# Flagged candidates statistics
					col1, col2, col3 = st.columns(3)
					with col1:
						st.metric("Flagged Count", len(flagged_df))
					with col2:
						st.metric("Avg Bias Score", f"{flagged_df['bias_score'].mean():.3f}")
					with col3:
						st.metric("Avg ATS Score", f"{flagged_df['ats_score'].mean():.1f}")
				else:
					st.success("✅ No candidates flagged above the selected threshold.")
		except Exception as e:
			st.error(f"Error in bias analysis: {e}")

		# Enhanced Domain Performance Deep Dive
		with st.expander("🔍 Domain Performance Deep Dive", expanded=False):
			try:
				df_performance = get_domain_performance_stats()
				if not df_performance.empty:
					st.markdown("#### Comprehensive Domain Performance Metrics")
					
					# Performance heatmap
					performance_cols = ['avg_ats_score', 'avg_edu_score', 'avg_exp_score', 
								'avg_skills_score', 'avg_lang_score', 'avg_keyword_score']
					
					if all(col in df_performance.columns for col in performance_cols):
						heatmap_data = df_performance[['domain'] + performance_cols].set_index('domain')
						
						fig = px.imshow(heatmap_data.T, 
									title="Domain Performance Heatmap",
									color_continuous_scale="RdYlGn",
									aspect="auto")
						fig.update_layout(height=400)
						st.plotly_chart(fig, use_container_width=True)
					
					# Detailed performance table
					st.dataframe(
						df_performance.style.format({
							col: '{:.2f}' for col in performance_cols + ['avg_bias_score']
						}),
						use_container_width=True
					)
				else:
					st.info("ℹ️ No detailed performance data available.")
			except Exception as e:
				st.error(f"Error loading performance deep dive: {e}")

		# Footer with system information
		st.markdown("<hr style='border-top: 1px solid #ddd; margin: 2rem 0;'>", unsafe_allow_html=True)
		st.markdown("""
		<div style='text-align: center; color: #666; font-size: 0.9em; padding: 1rem;'>
			<p>🛡️ Enhanced Admin Dashboard | Powered by Advanced Database Manager</p>
			<p>Last updated: {}</p>
		</div>
		""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)