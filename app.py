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

from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

def clear_existing_databases():
    """
    Clear existing ChromaDB and SQLite databases
    """
    # Remove folders with alphanumeric names
    for folder in glob.glob('[0-9a-f]*-*-*-*-*'):
        try:
            if os.path.isdir(folder):
                shutil.rmtree(folder)
        except Exception as e:
            st.warning(f"Could not remove folder {folder}: {e}")

    # Remove SQLite and Chroma files
    for file_pattern in ['*.sqlite3', 'chroma.sqlite3', 'vanna_user_database.sqlite']:
        for file in glob.glob(file_pattern):
            try:
                os.remove(file)
            except Exception as e:
                st.warning(f"Could not remove file {file}: {e}")

    # Clear temp directory ChromaDB and SQLite
    try:
        chroma_path = os.path.join(tempfile.gettempdir(), 'chroma')
        sqlite_path = os.path.join(tempfile.gettempdir(), 'vanna_user_database.sqlite')
        
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
        if os.path.exists(sqlite_path):
            os.remove(sqlite_path)
    except Exception as e:
        st.warning(f"Could not clear temp databases: {e}")

def safe_get_similar_questions(vn, prompt, sql, results_df):
    """
    Safely get similar questions, including SQL and results DataFrame
    """
    try:
        # Generate similar questions using the full context
        similar_questions = vn.generate_followup_questions(prompt, sql, results_df)
        
        # Ensure we're working with a list of questions
        if isinstance(similar_questions, list):
            # If list of dicts, extract questions
            if similar_questions and isinstance(similar_questions[0], dict):
                similar_questions = [q.get('question', '') for q in similar_questions if isinstance(q, dict)]
            
            # Remove any empty strings and duplicates
            similar_questions = list(dict.fromkeys(filter(bool, similar_questions)))
        else:
            # If not a list, convert to empty list
            similar_questions = []
        
        return similar_questions
    except Exception as e:
        st.warning(f"Error getting similar questions: {e}")
        return []

def main():
    # Perform one-time database cleanup
    if 'databases_cleared' not in st.session_state:
        clear_existing_databases()
        st.session_state.databases_cleared = True

    st.set_page_config(page_title="Talk2Table", layout="wide")
    st.title("🤖 Talk2Table")

    # Sidebar for configuration
    st.sidebar.header("OpenAI Configuration")
    openai_api_key = st.sidebar.text_input(label="OpenAI API KEY",placeholder="sk-...", type="password")
    
    # Configuration checkboxes
    show_sql = st.sidebar.checkbox("Show SQL Query", value=True)
    show_table = st.sidebar.checkbox("Show Data Table", value=True)
    show_chart = st.sidebar.checkbox("Show Plotly Chart", value=True)
    show_summary = st.sidebar.checkbox("Show Summary", value=False)

    # Initialize session state for chat messages and similar questions
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'similar_questions' not in st.session_state:
        st.session_state.similar_questions = []

    # CSV File Upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    # Chat container
    chat_container = st.container()

    if uploaded_file is not None and openai_api_key:
        # Unique key for this upload to prevent unnecessary reruns
        upload_key = hash(uploaded_file.name + openai_api_key)
        
        # Only process if this is a new upload
        if 'last_upload_key' not in st.session_state or st.session_state.last_upload_key != upload_key:
            # Save uploaded file temporarily and load to SQLite
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_csv:
                temp_csv.write(uploaded_file.getvalue())
                temp_csv_path = temp_csv.name

            # Load CSV to SQLite
            db_path, df = load_csv_to_sqlite(temp_csv_path)
            
            # Store the upload key to prevent reprocessing
            st.session_state.last_upload_key = upload_key
            
            if db_path and df is not None:
                # Create Vanna instance and connect to SQLite
                vn = MyVanna(config={'api_key': openai_api_key, 'model': 'gpt-3.5-turbo-0125'})
                vn.connect_to_sqlite(db_path)

                # Train Vanna with table schema
                df_information_schema = vn.run_sql("PRAGMA table_info('user_data');")
                plan_df = convert_to_information_schema_df(df_information_schema)
                
                plan = vn.get_training_plan_generic(plan_df)
                vn.train(plan=plan)

                # Store Vanna instance in session state
                st.session_state.vanna_instance = vn
                st.session_state.dataframe = df

        # Retrieve stored Vanna instance and dataframe
        vn = st.session_state.get('vanna_instance')
        df = st.session_state.get('dataframe')

        if vn and df is not None:
            # Display existing messages
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            # Sidebar for suggested questions
            st.sidebar.header("Suggested Questions")
            for q in st.session_state.similar_questions:
                st.sidebar.markdown("* "+q)

            # Initialize prompt to None at the start
            prompt = None

            # Chat input
            prompt = st.chat_input("Ask a question about your data...")

            # Process the query if a prompt exists
            if prompt:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Process query
                with st.chat_message("assistant"):
                    with st.spinner("Generating answer..."):
                        try:
                            # Use Vanna to generate answer
                            sql, results_df, fig = vn.ask(
                                question=prompt, 
                                print_results=False, 
                                auto_train=True, 
                                visualize=show_chart
                            )
                            

                            # Safely generate similar questions
                            similar_questions = []
                            try:
                                if results_df is not None:
                                    similar_questions = safe_get_similar_questions(vn, prompt, sql, results_df)
                                else:
                                    # Fallback if no results DataFrame
                                    similar_questions = vn.generate_followup_questions(prompt)
                            except Exception as sq_error:
                                st.warning(f"Could not generate similar questions: {sq_error}")

                            # Update similar questions in session state
                            st.session_state.similar_questions = similar_questions[-5:]

                            # Prepare response
                            response = ""
                            
                            # Add SQL details
                            if show_sql and sql:
                                response += f"**Generated SQL:**\n```sql\n{sql}\n```\n\n"
                            
                            # Add summary if enabled
                            if show_summary and results_df is not None:
                                try:
                                    summary = vn.generate_summary(prompt, results_df)
                                    response += f"**Summary:**\n{summary}\n\n"
                                except Exception as sum_error:
                                    st.warning(f"Could not generate summary: {sum_error}")
                            
                            # Add table output
                            if show_table and results_df is not None:
                                try:
                                    response += "**Data Results:**\n" + results_df.to_markdown() + "\n\n"
                                except Exception as table_error:
                                    st.warning(f"Could not display table: {table_error}")
                                    response += "**Data Results:** Unable to display table\n\n"

                            # Display or handle chart
                            if show_chart and fig is not None:
                                st.plotly_chart(fig, use_container_width=True)

                            # Display response
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})

                        except Exception as e:
                            error_message = f"Error generating answer: {str(e)}"
                            st.error(error_message)
                            st.session_state.messages.append({"role": "assistant", "content": error_message})

    else:
        st.info("Please provide both OpenAI API Key and upload a CSV file to enable chat.")

def load_csv_to_sqlite(csv_file, table_name='user_data'):
    db_path = os.path.join(tempfile.gettempdir(), 'vanna_user_database.sqlite')
    df = pd.read_csv(csv_file, encoding_errors='ignore')
    
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
            
    return db_path, df

def convert_to_information_schema_df(input_df):
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

if __name__ == "__main__":
    main()
