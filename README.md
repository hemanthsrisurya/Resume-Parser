# AI Resume Analyzer and LinkedIn Scraper using Generative AI

**Introduction**

Developed an advanced AI application that leverages Retrieval-Augmented Generation (RAG), Large Language Models (LLM), Groq API, and Nomic Embeddings for comprehensive resume analysis. It excels at summarizing the resume, evaluating strengths, identifying weaknesses, and offering personalized improvement suggestions while also recommending the perfect job titles. Additionally, it seamlessly employs Selenium to extract vital LinkedIn data, including company names, job titles, locations, job URLs, and detailed job descriptions. This application simplifies the job-seeking journey by equipping users with comprehensive insights to elevate their career opportunities.

<br />

**Table of Contents**

1. Key Technologies and Skills
2. Installation
3. Usage
4. Features
5. Contributing
6. License
7. Contact

<br />

**Key Technologies and Skills**
- Python
- NumPy
- Pandas
- LangChain
- Large Language Model (LLM)
- Retrieval-Augmented Generation (RAG)
- Groq API (Free)
- Nomic Embeddings (Free)
- Selenium
- AWS
- Hugging Face
- Streamlit

<br />

**Installation**

To run this project, you need to install the following packages:

```python
pip install numpy
pip install pandas
pip install streamlit
pip install streamlit_option_menu
pip install streamlit_extras
pip install PyPDF2
pip install langchain
pip install langchain-groq
pip install langchain-nomic
pip install tiktoken
pip install faiss-cpu
pip install selenium
```

<br />

**Usage**

To use this project, follow these steps:

1. Clone the repository: ```https://github.com/gopiashokan/AI-Powered-Resume-Analyzer-and-LinkedIn-Scraper-with-Selenium.git```
2. Install the required packages: ```pip install -r requirements.txt```
3. Get your free API keys:
   - Groq API: Sign up at [https://console.groq.com/](https://console.groq.com/)
   - Nomic API: Sign up at [https://atlas.nomic.ai/](https://atlas.nomic.ai/)
4. Run the Streamlit app: ```streamlit run app.py```
5. Access the app in your browser at ```http://localhost:8501```

<br />

**Features**

**Easy User Experience:**
- Resume Analyzer AI makes it easy for users. You can upload your resume and enter your Groq and Nomic API keys without any hassle. Both services offer free tiers, making this application accessible to everyone.
- It also uses the PyPDF2 library to quickly extract text from your uploaded resume, which is the first step in doing a thorough analysis.

**Smart Text Analysis with Langchain:**
- What makes it special is how it analyzes text. It uses a smart method called the Langchain library to break long sections of text from resumes into smaller chunks, making them more meaningful.
- This clever technique improves the accuracy of the resume analysis, and it gives users practical advice on how to enhance their job prospects.

**Enhanced Groq and Nomic Integration with FAISS:**
- Seamlessly connecting to Groq and Nomic services, the application establishes a secure connection using your API keys. This integration forms the basis for robust interactions, facilitating advanced analysis and efficient information retrieval.
- It uses the FAISS(Facebook AI Similarity Search) library to convert both the text chunks and query text data into numerical vectors, simplifying the analysis process and enabling the retrieval of pertinent information.

**Intelligent Chunk Selection in RAG and LLM:**
- The application retrieves relevant text by comparing the user query with stored vector embeddings and selecting the Top K documents based on their similarity scores. This ensures that only the most relevant information is considered for further processing.
- Once the most relevant documents are selected, the system initializes a Large Language Model (LLM), specifically the Llama 3.3 70B model via Groq to analyze and generate responses based on the retrieved content.

**Robust Question-Answering Pipeline:**
- The QA pipeline processes the Top K documents and the user query to generate meaningful responses. The system extracts relevant information from the retrieved content, ensuring accuracy and coherence.
- The LLM analyzes the selected documents and formulates responses by understanding the context within the retrieved text. This approach enhances response quality by focusing on the most relevant data instead of relying solely on the initial query.

**Comprehensive Resume Analysis:**
- **Summary:** Resume Analyzer AI provides a quick, comprehensive overview of resumes, emphasizing qualifications, key experience, skills, projects, and achievements. Users can swiftly grasp profiles, enhancing review efficiency and insight.
- **Strength:** Effortlessly conducting a comprehensive resume review, it analyzes qualifications, experience, and accomplishments. It subsequently highlights strengths, providing job seekers with a competitive edge.
- **Weakness:** AI conducts thorough analysis to pinpoint weaknesses and offers tailored solutions for transforming them into strengths, empowering job seekers.
- **Suggestion:** AI provides personalized job title recommendations that align closely with the user's qualifications and resume content, facilitating an optimized job search experience.

<br />

🚀 **Streamlit application:** [https://huggingface.co/spaces/gopiashokan/Resume-Analyzer-AI](https://huggingface.co/spaces/gopiashokan/Resume-Analyzer-AI)

<br />

**Selenium-Powered LinkedIn Data Scraping:**
- Utilizing Selenium and a Webdriver automated test tool, this feature enables users to input job titles, automating the data scraping process from LinkedIn. The scraped data includes crucial details such as company names, job titles, locations, URLs, and comprehensive job descriptions.
- This streamlined process enables users to easily review scraped job details and apply for positions, simplifying their job search and application experience.

<br />

🎥 **Project Demo Video:** [https://youtu.be/wFouWeK7NPg](https://youtu.be/wFouWeK7NPg)

<br />

**Contributing**

Contributions to this project are welcome! If you encounter any issues or have suggestions for improvements, please feel free to submit a pull request.

<br />

**License**

This project is licensed under the MIT License. Please review the LICENSE file for more details.

<br />

**Contact**

📧 Email: gopiashokankiot@gmail.com 

🌐 LinkedIn: [linkedin.com/in/gopiashokan](https://www.linkedin.com/in/gopiashokan)

For any further questions or inquiries, feel free to reach out. We are happy to assist you with any queries.

