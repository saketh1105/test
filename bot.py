

import urllib.request
import fitz
import re
import numpy as np
import tensorflow_hub as hub
import openai
import gradio as gr
import os
from sklearn.neighbors import NearestNeighbors

import warnings
OPENAI_API_KEY="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"

warnings.filterwarnings("ignore", message="The `style` method is deprecated.*")

def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page-1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []
    
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    
    def __init__(self):
        self.use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False
    
    
    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True
    
    
    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]
        
        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors
    
    
    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings

def load_recommender(path, start_page=1):
    global recommender
    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'

def generate_text(openAI_key, prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.8,
    )
    message = completions.choices[0].text
    return message

def generate_answer(question):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'
        
    #prompt += "Instructions: behave like you are a chatbot that answers questions based on the specified context relavant and answer the question elaborately from the given trained information exclusive for ISB pgp only. Only include information found in the results, if the search results mention multiple subjects don't add any additional information. Provide the linka whenever necessary from the provided document. Provide the links wheverev necessary where the context if followed by a URL provided in the document. Search results which have nothing to do with the question. Any response that mentions a fee figure, disregard it. Only answer what is asked,if the question can't be answered based on the ISB PGP document context generate sorry, I do not know'\n\nQuery: {question}\nAnswer: "
    prompt += "Perform a comprehensive analysis of innovation initiatives adopted by various firms and companies to address grand societal challenges through open innovation. Utilize data extracted from company annual reports to identify and evaluate the processes and capabilities employed by these firms. Specifically, assess the impact of these innovation initiatives on both societal betterment and the firm's revenue. consider the following keywords as indicators of innovation within the reports: platform, technology, application, program, projects, transformation, systems, digital, innovation, initiatives, solutions, impact, sustainability, society, and financial. Your analysis should include: Identification of firms and companies that have initiated innovation activities that may be any random name for each company related to grand societal challenges based on the presence of the specified keywords in their annual reports. A detailed description of the innovation initiatives undertaken by each firm, including the objectives, strategies, and technologies involved. Assessment of the impact of these initiatives on societal betterment, considering metrics such as improvements in quality of life, environmental sustainability, and social welfare. Evaluation of the impact of these initiatives on the firm's financial performance, including revenue growth, cost savings, and profitability. If anything asked out from the context answer I do noy know\n\nQuery: {question}\nAnswer: "
#               " "\
#               " "\
#               "with the same name, create separate answers for each. Only include information found in the results and "\
#               ""\
#               "If the text does not relate to the query, simply state 'Found Nothing'. Ignore outlier "\
#     prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
#               ". "\
#               "Citation should be done at the end of each sentence. If the search results mention multiple subjects "\
#               "with the same name, create separate answers for each. Only include information found in the results andPlease note that the query can be in any language and the sentence formation or spellings may be incorrect.  "\
#               "don't add any additional information. Make sure the answer is correct and don't output false content. "\
#               "If the text does not relate to the query, simply state 'Found Nothing'. Ignore outlier "\
#               "search results which have nothing to do with the question. Only answer what is asked. "\
#               "However, the response should aim to answer the query accurately.\n\nQuery: {question}\nAnswer: "

    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text(OPENAI_API_KEY, prompt, "text-davinci-003")
    return answer
def question_answer(url, file, question):

    if url.strip() != '' and file is not None:
        return '[ERROR]: Both URL and PDF are provided. Please provide only one (either URL or PDF).'

    if url.strip() != '':
        glob_url = url
        download_pdf(glob_url, 'corpus.pdf')
        load_recommender('corpus.pdf')

    else:
        file_name = file.name
        new_file_name = generate_unique_file_name(file_name)
        os.rename(file_name, new_file_name)
        load_recommender(new_file_name)

    if question.strip() == '':
        return '[ERROR]: Question field is empty'

    return generate_answer(question)

recommender = SemanticSearch()

def generate_unique_file_name(file_name):
    base_name, ext = os.path.splitext(file_name)
    i = 1
    while os.path.exists(file_name):
        file_name = f"{base_name}_{i}{ext}"
        i += 1
    return file_name

title = 'ISB PGP GPT'
with gr.Blocks() as demo:

    gr.Markdown(f'<center><h1>{title}</h1></center>')

    with gr.Row():
        
        with gr.Group():
            url = gr.Textbox(label='Enter PDF URL here')
            gr.Markdown("<center><h4>OR<h4></center>")
            file = gr.File(label='Upload your PDF here', file_types=['.pdf'])
            question = gr.Textbox(label='Enter your question here')
            btn = gr.Button(value='Submit')
            btn.style(scale=True)

        with gr.Group():
            answer = gr.Textbox(label='The answer to your question is :')

        btn.click(question_answer, inputs=[url, file, question], outputs=[answer])

demo.launch()


# In[ ]:




