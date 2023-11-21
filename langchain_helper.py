from langchain.llms import GooglePalm
import google.generativeai as genai
from langchain.utilities import SQLDatabase
from dotenv import load_dotenv
from langchain_experimental.sql import SQLDatabaseChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector, example_selector, FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX,_mysql_prompt
from langchain.prompts.prompt import PromptTemplate
import os
load_dotenv()
def few_shots_db_chain():
 llm=GooglePalm(google_api_key=os.environ["api_key"],temperature=0.1)
 db_user="root"
 db_password="Mayar17mayar"
 db_host="localhost"
 db_name="Chih3bTshirts"
 db=SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info=3)
 db_chain=SQLDatabaseChain.from_llm(llm,db,verbose=True)
 qns1 = db_chain.run("How many t-shirts do we have left for nike in extra small size and white color?")
 qns2 = db_chain.run("SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'")
 sql_code = """
 select sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
 (select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
 group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """
 qns3 = db_chain.run(sql_code)
 qns4 = db_chain.run("SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'")
 qns5 = db_chain.run("SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'")
 few_shots = [
     {'Question' : "How many t-shirts do we have left for Nike in XS size and white color?",
      'SQLQuery' : "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XS'",
      'SQLResult': "Result of the SQL query",
      'Answer' : qns1},
     {'Question': "How much is the total price of the inventory for all S-size t-shirts?",
      'SQLQuery':"SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'",
      'SQLResult': "Result of the SQL query",
      'Answer': qns2},
     {'Question': "If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?" ,
      'SQLQuery' : """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
 (select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
 group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
  """,
      'SQLResult': "Result of the SQL query",
      'Answer': qns3} ,
      {'Question' : "If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate without discount?" ,
       'SQLQuery': "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'",
       'SQLResult': "Result of the SQL query",
       'Answer' : qns4},
     {'Question': "How many white color Levi's shirt I have?",
      'SQLQuery' : "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'",
      'SQLResult': "Result of the SQL query",
      'Answer' : qns5
      }
 ]
 embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
 to_vectorize = [" ".join(example.values()) for example in few_shots]
 vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
 example_selector=SemanticSimilarityExampleSelector(
  vectorstore=vectorstore,
  k=2
 )
 example_prompt=PromptTemplate(input_variables=["Question","SQLQuery","SQLResult","Answer",],
                           template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer {Answer}")
 few_shots_prompt=FewShotPromptTemplate(
  example_selector=example_selector,
  example_prompt=example_prompt,
  prefix=_mysql_prompt,
  suffix=PROMPT_SUFFIX,
  input_variables=["input","table_info","top_k"]
 )
 new_chain=SQLDatabaseChain.from_llm(llm,db,verbose=True,prompt=few_shots_prompt)
 return new_chain
if __name__=="__main__":
 chain=few_shots_db_chain()