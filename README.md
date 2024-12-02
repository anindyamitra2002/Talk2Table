# Talk2Table  

## Overview  
**Talk2Table** is an intelligent Retrieval-Augmented Generation (RAG) system that allows users to interact with uploaded CSV data stored in an SQLite database. It uses OpenAI's GPT-based models for generating SQL queries and analyzing data, enabling users to ask natural language questions and receive comprehensive answers, including SQL queries, data tables, summaries, and visualizations.  

With Talk2Table, you can effortlessly extract insights from your data without writing any SQL or complex scripts.  

---

## Contents  
- [Overview](#overview)  
- [Contents](#contents)  
- [How It Works](#how-it-works)  
- [Features](#features)  
- [Getting Started](#getting-started)  
- [Why Talk2Table](#why-talk2table)  
- [Acknowledgement](#acknowledgement)  

---

## How It Works  

### 1. Upload Data  
Users can upload their data in the form of a CSV file. Once uploaded, the data is automatically imported into an SQLite database, ensuring quick and efficient access for querying and analysis.  

### 2. Query with Natural Language  
Simply type your questions in plain English. The system uses OpenAI’s GPT-based models to interpret your query and generate a corresponding SQL command.  

### 3. Retrieve Table Data  
The SQL query is executed on the uploaded data, and the retrieved results are displayed in tabular format. You can view the specific rows or aggregated data that answer your query.  

### 4. Visualization  
The retrieved data can also be visualized as interactive charts using Plotly. This allows users to gain deeper insights into their data with clear and visually appealing representations.  

### 5. Similar Questions  
The system suggests follow-up questions based on the context of your query and retrieved data. This feature encourages exploratory data analysis and helps users uncover additional insights from their data.  

---

## Features  

### Similar Questions  
The system generates follow-up questions to guide users toward exploring their data more effectively. This feature ensures a conversational and intuitive experience for data interaction.  

### Plotly Visualizations  
Data can be displayed as interactive charts and graphs, enabling users to analyze trends, patterns, and insights visually.  

### Table Retrieval  
Query results are displayed as structured data tables, providing precise and actionable insights. Users can easily explore their dataset row by row or in aggregated form.  

### Natural Language to SQL Generation  
The system translates natural language queries into SQL commands automatically, removing the need for SQL knowledge and making data querying accessible to everyone.  

### Configurable Outputs  
Users can choose what to display, such as SQL queries, tables, visualizations, and summaries, to tailor the output to their needs.  

### Easy Cleanup  
Automatically clears previous databases and ensures a clean workspace for every session.  

---

## Getting Started  

### Prerequisites  
- Python 3.10 or higher  
- OpenAI API key (Sign up at [OpenAI](https://platform.openai.com/signup/) to obtain one)  

### Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/Talk2Table.git
   cd Talk2Table
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

### Running the Application  
1. Start the Streamlit app:  
   ```bash
   streamlit run main.py
   ```  
2. Open the application in your browser (usually at [http://localhost:8501](http://localhost:8501)).  
3. Upload a CSV file and enter your OpenAI API key in the sidebar.  

---

## Why Talk2Table  

- **Ease of Use**: No prior SQL knowledge is required to query your data.  
- **High Accuracy on Complex Datasets**: Leverages state-of-the-art GPT-based models to generate highly accurate SQL queries, even for intricate datasets.  
- **Secure and Private**: Ensures the safety and privacy of your data by processing everything locally and keeping user data secure.  
- **Powerful Insights**: Provides actionable insights with SQL-generated results, summaries, and visualizations.  
- **Conversational Experience**: Engages users with suggested follow-up questions to encourage exploratory data analysis.  
- **Customizable Outputs**: Tailor the displayed information based on your needs (SQL, tables, charts, summaries).  
- **Flexible Deployment**: Streamlit-based app can run locally or be deployed on cloud platforms.  

---

## Acknowledgement  
We extend our gratitude to the following tools and technologies:  
- **[OpenAI](https://openai.com/)**: For GPT models powering the natural language understanding and SQL generation.  
- **[Streamlit](https://streamlit.io/)**: For providing an intuitive framework to create interactive web applications.  
- **[SQLite](https://www.sqlite.org/)**: For the robust database management system.  
- **[Plotly](https://plotly.com/)**: For enabling dynamic and interactive data visualizations.  
- **[Vanna AI](https://www.vanna.ai/)**: For its exceptional integration of OpenAI and ChromaDB, enhancing RAG systems like Talk2Table.  

Thank you for using **Talk2Table**! If you have any suggestions or feedback, feel free to contribute to the project or raise issues on GitHub.  
