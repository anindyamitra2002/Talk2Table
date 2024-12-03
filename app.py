import os
import sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import pandas as pd
import tempfile
import shutil
import glob
import plotly.graph_objs as go
import plotly.io as pio
import json

from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create temp directories in the script's parent directory
        temp_dir = os.path.join(script_dir, 'temp_talk2table')
        os.makedirs(temp_dir, exist_ok=True)
        
        # ChromaDB path
        chroma_path = os.path.join(temp_dir, 'chromadb')
        
        # Update config with local paths
        if config is None:
            config = {}
        config['persist_directory'] = chroma_path
        
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

def clear_existing_databases():
    """
    Clear existing temporary databases and directories
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, 'temp_talk2table')
    
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            st.success("Temporary databases and directories cleared successfully.")
        except Exception as e:
            st.error(f"Error clearing databases: {e}")
    else:
        st.info("No temporary databases found.")

@st.cache_resource(ttl=3600)
def setup_vanna(openai_api_key):
    """
    Set up Vanna instance with caching to prevent recreation on every rerun
    """
    vn = MyVanna(config={
        'api_key': openai_api_key, 
        'model': 'gpt-3.5-turbo-0125',
        'allow_llm_to_see_data': True
    })
    return vn

@st.cache_data(ttl=3600)
def load_csv_to_sqlite(csv_file, table_name='user_data'):
    """
    Cache the CSV to SQLite conversion with local temp directory
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, 'temp_talk2table')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create SQLite database in the temp directory
    db_path = os.path.join(temp_dir, 'vanna_user_database.sqlite')
    
    df = pd.read_csv(csv_file, encoding_errors='ignore')
    
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
            
    return db_path, df

@st.cache_data(ttl=3600)
def convert_to_information_schema_df(input_df):
    """
    Convert input DataFrame to information schema DataFrame
    """
    rows = []
    database = 'main'
    schema = 'public'
    table_name = 'user_data'
    
    for _, row in input_df.iterrows():
        row_data = {
            'TABLE_CATALOG': database,
            'TABLE_SCHEMA': schema,
            'TABLE_NAME': table_name,
            'COLUMN_NAME': row['name'],
            'DATA_TYPE': row['type'],
            'IS_NULLABLE': 'NO' if row['notnull'] else 'YES',
            'COLUMN_DEFAULT': row['dflt_value'],
            'IS_PRIMARY_KEY': 'YES' if row['pk'] else 'NO'
        }
        rows.append(row_data)
    
    return pd.DataFrame(rows)

def generate_followup_questions_cached(vn, prompt, sql=None, df=None):
    """
    Safely generate follow-up questions with optional SQL and DataFrame
    """
    try:
        # If both SQL and DataFrame are provided, use the method that requires them
        if sql is not None and df is not None:
            similar_questions = vn.generate_followup_questions(prompt, sql, df)
        else:
            # Fallback to method without SQL and DataFrame
            similar_questions = vn.generate_followup_questions(prompt)
        
        # Ensure we're working with a list of questions
        if isinstance(similar_questions, list):
            # If list of dicts, extract questions
            if similar_questions and isinstance(similar_questions[0], dict):
                similar_questions = [q.get('question', '') for q in similar_questions if isinstance(q, dict)]
            
            # Remove empty strings and duplicates
            similar_questions = list(dict.fromkeys(filter(bool, similar_questions)))
        else:
            similar_questions = []
        
        return similar_questions[:5]  # Limit to 5 follow-up questions
    except Exception as e:
        st.warning(f"Error getting similar questions: {e}")
        return []

def main():
    st.set_page_config(page_title="Talk2Table", layout="wide")
    st.title("🤖 Talk2Table")

    # Sidebar for configuration
    st.sidebar.header("OpenAI Configuration")
    openai_api_key = st.sidebar.text_input(label="OpenAI API KEY", placeholder="sk-...", type="password")
    
    # Add a button to clear existing databases
    if st.sidebar.button("Clear Temp Databases"):
        clear_existing_databases()
    
    # Configuration checkboxes
    show_sql = st.sidebar.checkbox("Show SQL Query", value=True)
    show_table = st.sidebar.checkbox("Show Data Table", value=True)
    show_chart = st.sidebar.checkbox("Show Plotly Chart", value=True)
    show_summary = st.sidebar.checkbox("Show Summary", value=False)

    # Initialize or reset session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # CSV File Upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    # Chat container
    chat_container = st.container()

    if uploaded_file is not None and openai_api_key:
        # Save uploaded file temporarily and load to SQLite
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(script_dir, 'temp_talk2table')
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_csv_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_csv_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Load CSV to SQLite
        db_path, df = load_csv_to_sqlite(temp_csv_path)
        
        if db_path and df is not None:
            # Setup Vanna instance with caching
            vn = setup_vanna(openai_api_key)
            
            # Connect to SQLite and train
            vn.connect_to_sqlite(db_path)

            # Train Vanna with table schema
            df_information_schema = vn.run_sql("PRAGMA table_info('user_data');")
            plan_df = convert_to_information_schema_df(df_information_schema)
            
            # Enhanced training
            plan = vn.get_training_plan_generic(plan_df)
            vn.train(plan=plan)

            # Display existing messages and their plots
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        
                        # If the message has a plot, display it
                        if message["role"] == "assistant" and 'plot' in message:
                            try:
                                # Use plotly.io to parse the JSON figure
                                plot_fig = pio.from_json(message['plot'])
                                st.plotly_chart(plot_fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error rendering plot: {e}")

            # Sidebar for suggested questions
            st.sidebar.header("Suggested Questions")
            for q in st.session_state.get('similar_questions', []):
                st.sidebar.markdown("* "+q)

            prompt = st.chat_input("Ask a question about your data...")

            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Generating answer..."):
                        try:
                            # Generate SQL with explicit allow_llm_to_see_data
                            sql, results_df, fig = vn.ask(
                                question=prompt, 
                                print_results=False, 
                                auto_train=True, 
                                visualize=show_chart,
                                allow_llm_to_see_data=True
                            )

                            # Prepare response
                            response = ""
                            
                            # Prepare message with plot
                            assistant_message = {
                                "role": "assistant",
                                "content": "",
                                "plot": None
                            }
                            
                            # Update last successful query state
                            if sql:
                                st.session_state.last_prompt = prompt
                                st.session_state.last_sql = sql
                                st.session_state.last_df = results_df

                            if show_sql and sql:
                                response += f"**Generated SQL:**\n```sql\n{sql}\n```\n\n"
                            
                            if show_summary and results_df is not None:
                                try:
                                    summary = vn.generate_summary(prompt, results_df)
                                    response += f"**Summary:**\n{summary}\n\n"
                                except Exception as sum_error:
                                    st.warning(f"Could not generate summary: {sum_error}")
                            
                            if show_table and results_df is not None:
                                try:
                                    response += "**Data Results:**\n" + results_df.to_markdown() + "\n\n"
                                except Exception as table_error:
                                    st.warning(f"Could not display table: {table_error}")
                                    response += "**Data Results:** Unable to display table\n\n"

                            # Store the plot in the message
                            if show_chart and fig is not None:
                                # Use plotly.io to convert figure to JSON
                                assistant_message['plot'] = pio.to_json(fig, remove_uids=True)
                                st.plotly_chart(fig, use_container_width=True)

                            # Generate follow-up questions
                            similar_questions = generate_followup_questions_cached(
                                vn, 
                                prompt, 
                                sql=st.session_state.get('last_sql'), 
                                df=st.session_state.get('last_df')
                            )
                            st.session_state.similar_questions = similar_questions

                            # Finalize the assistant message
                            assistant_message['content'] = response
                            st.session_state.messages.append(assistant_message)

                            st.markdown(response)

                        except Exception as e:
                            error_message = f"Error generating answer: {str(e)}"
                            st.error(error_message)
                            st.session_state.messages.append({"role": "assistant", "content": error_message})

    else:
        st.info("Please provide both OpenAI API Key and upload a CSV file to enable chat.")

if __name__ == "__main__":
    main()