import time
import os
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()


# ===== SESSION STATE INITIALIZATION FOR PERSISTENCE =====
# This ensures embeddings are created once and reused across all sections
if 'pdf_chunks' not in st.session_state:
    st.session_state.pdf_chunks = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = os.getenv('GROQ_API_KEY', '')
if 'nomic_api_key' not in st.session_state:
    st.session_state.nomic_api_key = os.getenv('NOMIC_API_KEY', '')
if 'summary_analysis' not in st.session_state:
    st.session_state.summary_analysis = None
if 'strength_analysis' not in st.session_state:
    st.session_state.strength_analysis = None
if 'weakness_analysis' not in st.session_state:
    st.session_state.weakness_analysis = None
if 'job_titles_analysis' not in st.session_state:
    st.session_state.job_titles_analysis = None
if 'summary_stream_pending' not in st.session_state:
    st.session_state.summary_stream_pending = False
if 'strength_stream_pending' not in st.session_state:
    st.session_state.strength_stream_pending = False
if 'weakness_stream_pending' not in st.session_state:
    st.session_state.weakness_stream_pending = False
if 'job_titles_stream_pending' not in st.session_state:
    st.session_state.job_titles_stream_pending = False


def stream_text_output(text: str, delay: float = 0.02, target=None) -> None:
    """Stream text to the UI with a simple typewriter effect."""

    def _stream():
        for word in text.split(" "):
            yield word + " "
            time.sleep(delay)
        yield "\n"

    (target or st).write_stream(_stream)


def streamlit_config():

    # page configuration
    st.set_page_config(page_title='Resume Analyzer AI', layout="wide")

    # page header transparent color
    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    # title and position
    st.markdown(f'<h1 style="text-align: center;">Resume Analyzer AI</h1>',
                unsafe_allow_html=True)


class resume_analyzer:

    def pdf_to_chunks(pdf):
        # read pdf and it returns memory address
        pdf_reader = PdfReader(pdf)

        # extrat text from each page separately
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split the long text into small chunks.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=200,
            length_function=len)

        chunks = text_splitter.split_text(text=text)
        return chunks


    def openai(groq_api_key, nomic_api_key, chunks, analyze):

        # Set API keys in environment for Nomic (it reads from env automatically)
        os.environ['NOMIC_API_KEY'] = nomic_api_key
        os.environ['GROQ_API_KEY'] = groq_api_key

        # Using Nomic service for embedding
        embeddings = NomicEmbeddings(
            model="nomic-embed-text-v1.5",
            inference_mode="remote"
        )

        # Facebook AI Similarity Serach library help us to convert text data to numerical vector
        vectorstores = FAISS.from_texts(chunks, embedding=embeddings)

        # compares the query and chunks, enabling the selection of the top 'K' most similar chunks based on their similarity scores.
        docs = vectorstores.similarity_search(query=analyze, k=3)

        # creates a ChatGroq object for inference with Llama 4 Scout
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.6,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
            streaming=True,
            api_key=groq_api_key
        )

        # question-answering (QA) pipeline, making use of the load_qa_chain function
        chain = load_qa_chain(llm=llm, chain_type='stuff')

        response = chain.run(input_documents=docs, question=analyze)
        return response


    def create_vectorstore(nomic_api_key, chunks):
        """Create vector store once and cache it in session state"""
        # Set API key in environment for Nomic
        os.environ['NOMIC_API_KEY'] = nomic_api_key

        # Using Nomic service for embedding
        embeddings = NomicEmbeddings(
            model="nomic-embed-text-v1.5",
            inference_mode="remote"
        )

        # Create FAISS vector store
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        return vectorstore


    def query_llm(groq_api_key, vectorstore, analyze):
        """Query the LLM using existing vector store from session state"""
        # Set API key in environment
        os.environ['GROQ_API_KEY'] = groq_api_key

        # Get relevant documents from vector store
        docs = vectorstore.similarity_search(query=analyze, k=3)

        # creates a ChatGroq object for inference with Llama 4 Scout
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.6,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
            streaming=True,
            api_key=groq_api_key
        )

        # question-answering (QA) pipeline
        chain = load_qa_chain(llm=llm, chain_type='stuff')

        response = chain.run(input_documents=docs, question=analyze)
        return response


    def summary_prompt(query_with_chunks):

        query = f''' need to detailed summarization of below resume and finally conclude them

                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    def resume_summary():

        st.markdown('<h3 style="color: orange;">Upload & Analyze Resume</h3>', unsafe_allow_html=True)
        st.write("Make a Deep Analysis of your Resume.")

        with st.form(key='Summary'):

            # User Upload the Resume
            add_vertical_space(1)
            pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
            add_vertical_space(1)
            st.caption('Set your Groq and Nomic API keys from the sidebar before processing.')
            add_vertical_space(1)

            # Click on Submit Button
            submit = st.form_submit_button(label='Process Resume')
            add_vertical_space(1)

        add_vertical_space(3)

        groq_api_key = st.session_state.groq_api_key
        nomic_api_key = st.session_state.nomic_api_key

        if submit:
            if pdf is not None and groq_api_key != '' and nomic_api_key != '':
                try:
                    with st.spinner('Processing your resume....'):

                        # Extract chunks from the uploaded PDF
                        pdf_chunks = resume_analyzer.pdf_to_chunks(pdf)

                        # Create vector store and cache it
                        vectorstore = resume_analyzer.create_vectorstore(nomic_api_key, pdf_chunks)

                        # Store in session state for reuse
                        st.session_state.pdf_chunks = pdf_chunks
                        st.session_state.vectorstore = vectorstore
                        st.session_state.groq_api_key = groq_api_key
                        st.session_state.nomic_api_key = nomic_api_key

                        # Clear previous analysis results when new resume is uploaded
                        st.session_state.summary_analysis = None
                        st.session_state.strength_analysis = None
                        st.session_state.weakness_analysis = None
                        st.session_state.job_titles_analysis = None
                        st.session_state.summary_stream_pending = False
                        st.session_state.strength_stream_pending = False
                        st.session_state.weakness_stream_pending = False
                        st.session_state.job_titles_stream_pending = False

                        # Generate summary
                        summary_prompt = resume_analyzer.summary_prompt(pdf_chunks)
                        summary = resume_analyzer.query_llm(groq_api_key, vectorstore, summary_prompt)

                        # Cache summary result
                        st.session_state.summary_analysis = summary
                        st.session_state.summary_stream_pending = True

                    st.success('Resume processed successfully.')

                except Exception as e:
                    st.error(f'Error: {e}')

            elif pdf is None:
                st.warning('Please upload your resume.')

            elif groq_api_key == '' or nomic_api_key == '':
                st.warning('Please add both API keys in the sidebar before processing.')

        if st.session_state.summary_analysis:
            st.markdown(f'<h4 style="color: orange;">Summary:</h4>', unsafe_allow_html=True)
            summary_placeholder = st.empty()

            if st.session_state.summary_stream_pending:
                stream_text_output(st.session_state.summary_analysis, target=summary_placeholder)
                st.session_state.summary_stream_pending = False
            else:
                summary_placeholder.write(st.session_state.summary_analysis)


    def strength_prompt(query_with_chunks):
        query = f'''need to detailed analysis and explain of the strength of below resume and finally conclude them
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    def resume_strength():

        st.markdown('<h3 style="color: orange;">Resume Strengths</h3>', unsafe_allow_html=True)

        # Check if embeddings exist
        if st.session_state.vectorstore is None:
            st.warning('Please upload and process your resume first in the "Summary" tab.')
            return

        if st.button('Analyze Strengths', key='strength_btn'):
            try:
                with st.spinner('Analyzing strengths...'):

                    # Then analyze strengths
                    context_text = st.session_state.summary_analysis or "\n".join(st.session_state.pdf_chunks or [])
                    strength_prompt = resume_analyzer.strength_prompt(context_text)
                    strength = resume_analyzer.query_llm(
                        st.session_state.groq_api_key,
                        st.session_state.vectorstore,
                        strength_prompt
                    )

                st.session_state.strength_analysis = strength
                st.session_state.strength_stream_pending = True

            except Exception as e:
                st.error(f'Error: {e}')

        if st.session_state.strength_analysis:
            st.markdown(f'<h4 style="color: orange;">Strengths:</h4>', unsafe_allow_html=True)
            strength_placeholder = st.empty()

            if st.session_state.strength_stream_pending:
                stream_text_output(st.session_state.strength_analysis, target=strength_placeholder)
                st.session_state.strength_stream_pending = False
            else:
                strength_placeholder.write(st.session_state.strength_analysis)


    def weakness_prompt(query_with_chunks):
        query = f'''need to detailed analysis and explain of the weakness of below resume and how to improve make a better resume.

                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    def resume_weakness():

        st.markdown('<h3 style="color: orange;">Resume Weaknesses & Suggestions</h3>', unsafe_allow_html=True)

        # Check if embeddings exist
        if st.session_state.vectorstore is None:
            st.warning('Please upload and process your resume first in the "Summary" tab.')
            return

        if st.button('Identify Weaknesses', key='weakness_btn'):
            try:
                with st.spinner('Identifying weaknesses...'):

                    # Then analyze weaknesses
                    context_text = st.session_state.summary_analysis or "\n".join(st.session_state.pdf_chunks or [])
                    weakness_prompt = resume_analyzer.weakness_prompt(context_text)
                    weakness = resume_analyzer.query_llm(
                        st.session_state.groq_api_key,
                        st.session_state.vectorstore,
                        weakness_prompt
                    )

                st.session_state.weakness_analysis = weakness
                st.session_state.weakness_stream_pending = True

            except Exception as e:
                st.error(f'Error: {e}')

        if st.session_state.weakness_analysis:
            st.markdown(f'<h4 style="color: orange;">Weaknesses and Suggestions:</h4>', unsafe_allow_html=True)
            weakness_placeholder = st.empty()

            if st.session_state.weakness_stream_pending:
                stream_text_output(st.session_state.weakness_analysis, target=weakness_placeholder)
                st.session_state.weakness_stream_pending = False
            else:
                weakness_placeholder.write(st.session_state.weakness_analysis)


    def job_title_prompt(query_with_chunks):

        query = f''' what are the job roles i apply to likedin based on below?
                    
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    def job_title_suggestion():

        st.markdown('<h3 style="color: orange;">Job Title Suggestions</h3>', unsafe_allow_html=True)

        # Check if embeddings exist
        if st.session_state.vectorstore is None:
            st.warning('Please upload and process your resume first in the "Summary" tab.')
            return

        if st.button('Get Job Suggestions', key='job_btn'):
            try:
                with st.spinner('Generating job suggestions...'):

                    # Then get job suggestions
                    context_text = st.session_state.summary_analysis or "\n".join(st.session_state.pdf_chunks or [])
                    job_title_prompt = resume_analyzer.job_title_prompt(context_text)
                    job_title = resume_analyzer.query_llm(
                        st.session_state.groq_api_key,
                        st.session_state.vectorstore,
                        job_title_prompt
                    )

                st.session_state.job_titles_analysis = job_title
                st.session_state.job_titles_stream_pending = True

            except Exception as e:
                st.error(f'Error: {e}')

        if st.session_state.job_titles_analysis:
            st.markdown(f'<h4 style="color: orange;">Job Titles:</h4>', unsafe_allow_html=True)
            jobs_placeholder = st.empty()

            if st.session_state.job_titles_stream_pending:
                stream_text_output(st.session_state.job_titles_analysis, target=jobs_placeholder)
                st.session_state.job_titles_stream_pending = False
            else:
                jobs_placeholder.write(st.session_state.job_titles_analysis)

class linkedin_scraper:

    def webdriver_setup():
            
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        driver = webdriver.Chrome(options=options)
        driver.maximize_window()
        return driver


    def get_userinput():

        add_vertical_space(2)
        with st.form(key='linkedin_scarp'):

            add_vertical_space(1)
            col1,col2,col3 = st.columns([0.5,0.3,0.2], gap='medium')
            with col1:
                job_title_input = st.text_input(label='Job Title')
                job_title_input = job_title_input.split(',')
            with col2:
                job_location = st.text_input(label='Job Location', value='India')
            with col3:
                job_count = st.number_input(label='Job Count', min_value=1, value=1, step=1)

            # Submit Button
            add_vertical_space(1)
            submit = st.form_submit_button(label='Submit')
            add_vertical_space(1)
        
        return job_title_input, job_location, job_count, submit


    def build_url(job_title, job_location):

        b = []
        for i in job_title:
            x = i.split()
            y = '%20'.join(x)
            b.append(y)

        job_title = '%2C%20'.join(b)
        link = f"https://in.linkedin.com/jobs/search?keywords={job_title}&location={job_location}&locationId=&geoId=102713980&f_TPR=r604800&position=1&pageNum=0"

        return link
    

    def open_link(driver, link):

        while True:
            # Break the Loop if the Element is Found, Indicating the Page Loaded Correctly
            try:
                driver.get(link)
                driver.implicitly_wait(5)
                time.sleep(3)
                driver.find_element(by=By.CSS_SELECTOR, value='span.switcher-tabs__placeholder-text.m-auto')
                return
            
            # Retry Loading the Page
            except NoSuchElementException:
                continue


    def link_open_scrolldown(driver, link, job_count):
        
        # Open the Link in LinkedIn
        linkedin_scraper.open_link(driver, link)

        # Scroll Down the Page
        for i in range(0,job_count):

            # Simulate clicking the Page Up button
            body = driver.find_element(by=By.TAG_NAME, value='body')
            body.send_keys(Keys.PAGE_UP)

            # Locate the sign-in modal dialog 
            try:
                driver.find_element(by=By.CSS_SELECTOR, 
                                value="button[data-tracking-control-name='public_jobs_contextual-sign-in-modal_modal_dismiss']>icon>svg").click()
            except:
                pass

            # Scoll down the Page to End
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            driver.implicitly_wait(2)

            # Click on See More Jobs Button if Present
            try:
                x = driver.find_element(by=By.CSS_SELECTOR, value="button[aria-label='See more jobs']").click()
                driver.implicitly_wait(5)
            except:
                pass


    def job_title_filter(scrap_job_title, user_job_title_input):
        
        # User Job Title Convert into Lower Case
        user_input = [i.lower().strip() for i in user_job_title_input]

        # scraped Job Title Convert into Lower Case
        scrap_title = [i.lower().strip() for i in [scrap_job_title]]

        # Verify Any User Job Title in the scraped Job Title
        confirmation_count = 0
        for i in user_input:
            if all(j in scrap_title[0] for j in i.split()):
                confirmation_count += 1

        # Return Job Title if confirmation_count greater than 0 else return NaN
        if confirmation_count > 0:
            return scrap_job_title
        else:
            return np.nan


    def scrap_company_data(driver, job_title_input, job_location):

        # scraping the Company Data
        company = driver.find_elements(by=By.CSS_SELECTOR, value='h4[class="base-search-card__subtitle"]')
        company_name = [i.text for i in company]

        location = driver.find_elements(by=By.CSS_SELECTOR, value='span[class="job-search-card__location"]')
        company_location = [i.text for i in location]

        title = driver.find_elements(by=By.CSS_SELECTOR, value='h3[class="base-search-card__title"]')
        job_title = [i.text for i in title]

        url = driver.find_elements(by=By.XPATH, value='//a[contains(@href, "/jobs/")]')
        website_url = [i.get_attribute('href') for i in url]

        # combine the all data to single dataframe
        df = pd.DataFrame(company_name, columns=['Company Name'])
        df['Job Title'] = pd.DataFrame(job_title)
        df['Location'] = pd.DataFrame(company_location)
        df['Website URL'] = pd.DataFrame(website_url)

        # Return Job Title if there are more than 1 matched word else return NaN
        df['Job Title'] = df['Job Title'].apply(lambda x: linkedin_scraper.job_title_filter(x, job_title_input))

        # Return Location if User Job Location in Scraped Location else return NaN
        df['Location'] = df['Location'].apply(lambda x: x if job_location.lower() in x.lower() else np.nan)
        
        # Drop Null Values and Reset Index
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)

        return df 
        

    def scrap_job_description(driver, df, job_count):
        
        # Get URL into List
        website_url = df['Website URL'].tolist()
        
        # Scrap the Job Description
        job_description = []
        description_count = 0

        for i in range(0, len(website_url)):
            try:
                # Open the Link in LinkedIn
                linkedin_scraper.open_link(driver, website_url[i])

                # Click on Show More Button
                driver.find_element(by=By.CSS_SELECTOR, value='button[data-tracking-control-name="public_jobs_show-more-html-btn"]').click()
                driver.implicitly_wait(5)
                time.sleep(1)

                # Get Job Description
                description = driver.find_elements(by=By.CSS_SELECTOR, value='div[class="show-more-less-html__markup relative overflow-hidden"]')
                data = [i.text for i in description][0]
                
                # Check Description length and Duplicate
                if len(data.strip()) > 0 and data not in job_description:
                    job_description.append(data)
                    description_count += 1
                else:
                    job_description.append('Description Not Available')
            
            # If any unexpected issue 
            except:
                job_description.append('Description Not Available')
            
            # Check Description Count reach User Job Count
            if description_count == job_count:
                break

        # Filter the Job Description
        df = df.iloc[:len(job_description), :]

        # Add Job Description in Dataframe
        df['Job Description'] = pd.DataFrame(job_description, columns=['Description'])
        df['Job Description'] = df['Job Description'].apply(lambda x: np.nan if x=='Description Not Available' else x)
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        return df


    def display_data_userinterface(df_final):

        # Display the Data in User Interface
        add_vertical_space(1)
        if len(df_final) > 0:
            for i in range(0, len(df_final)):
                
                st.markdown(f'<h3 style="color: orange;">Job Posting Details : {i+1}</h3>', unsafe_allow_html=True)
                st.write(f"Company Name : {df_final.iloc[i,0]}")
                st.write(f"Job Title    : {df_final.iloc[i,1]}")
                st.write(f"Location     : {df_final.iloc[i,2]}")
                st.write(f"Website URL  : {df_final.iloc[i,3]}")

                with st.expander(label='Job Desription'):
                    st.write(df_final.iloc[i, 4])
                add_vertical_space(3)
        
        else:
            st.markdown(f'<h5 style="text-align: center;color: orange;">No Matching Jobs Found</h5>', 
                                unsafe_allow_html=True)


    def main():
        
        # Initially set driver to None
        driver = None
        
        try:
            job_title_input, job_location, job_count, submit = linkedin_scraper.get_userinput()
            add_vertical_space(2)
            
            if submit:
                if job_title_input != [] and job_location != '':
                    
                    with st.spinner('Chrome Webdriver Setup Initializing...'):
                        driver = linkedin_scraper.webdriver_setup()
                                       
                    with st.spinner('Loading More Job Listings...'):

                        # build URL based on User Job Title Input
                        link = linkedin_scraper.build_url(job_title_input, job_location)

                        # Open the Link in LinkedIn and Scroll Down the Page
                        linkedin_scraper.link_open_scrolldown(driver, link, job_count)

                    with st.spinner('scraping Job Details...'):

                        # Scraping the Company Name, Location, Job Title and URL Data
                        df = linkedin_scraper.scrap_company_data(driver, job_title_input, job_location)

                        # Scraping the Job Descriptin Data
                        df_final = linkedin_scraper. scrap_job_description(driver, df, job_count)
                    
                    # Display the Data in User Interface
                    linkedin_scraper.display_data_userinterface(df_final)

                
                # If User Click Submit Button and Job Title is Empty
                elif job_title_input == []:
                    st.markdown(f'<h5 style="text-align: center;color: orange;">Job Title is Empty</h5>', 
                                unsafe_allow_html=True)
                
                elif job_location == '':
                    st.markdown(f'<h5 style="text-align: center;color: orange;">Job Location is Empty</h5>', 
                                unsafe_allow_html=True)

        except Exception as e:
            add_vertical_space(2)
            st.markdown(f'<h5 style="text-align: center;color: orange;">{e}</h5>', unsafe_allow_html=True)
        
        finally:
            if driver:
                driver.quit()



# Streamlit Configuration Setup
streamlit_config()
add_vertical_space(2)



with st.sidebar:

    st.markdown('### API Keys')
    add_vertical_space(1)
    st.session_state.groq_api_key = st.text_input(
        label='Groq API Key',
        value=st.session_state.groq_api_key,
        type='password',
        help='Stored in session state for this browser tab.'
    )
    st.session_state.nomic_api_key = st.text_input(
        label='Nomic API Key',
        value=st.session_state.nomic_api_key,
        type='password',
        help='Stored in session state for this browser tab.'
    )

    add_vertical_space(2)
    st.markdown('### Navigation')

    option = option_menu(menu_title='', options=['Summary', 'Strength', 'Weakness', 'Job Titles', 'Linkedin Jobs'],
                         icons=['house-fill', 'database-fill', 'pass-fill', 'list-ul', 'linkedin'])



if option == 'Summary':

    resume_analyzer.resume_summary()



elif option == 'Strength':

    resume_analyzer.resume_strength()



elif option == 'Weakness':

    resume_analyzer.resume_weakness()



elif option == 'Job Titles':

    resume_analyzer.job_title_suggestion()



elif option == 'Linkedin Jobs':
    
    linkedin_scraper.main()


