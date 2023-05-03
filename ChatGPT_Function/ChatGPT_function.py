import os, sys
import openai
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import re
from dotenv import load_dotenv
from embedding_parser import embedding_parser
from pdf_parser import Parser
from kiwipiepy import Kiwi



load_dotenv('/home/aift-ml/workspace/Langchain/.env')
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class gpt_func:
    def __init__(self):
        self.enc35 = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.api_key ="chatgpt api key"
        self.ep = embedding_parser()
        self.ps = Parser()
        self.kiwi = Kiwi()
        
    def split_into_sentence(self, text):
        temp = list()
        splits = self.kiwi.split_into_sents(text)
        for sentence in splits:
            temp.append(sentence[0])
        split_text = "\n".join(temp)   
        return split_text
        
        
    def token_length(self,input_text):
        tokenized_text_chatgpt35 = self.enc35.encode(input_text)
        # print(len(tokenized_text_chatgpt35))
        return len(tokenized_text_chatgpt35)
   
   
    def basic_gpt_api(self, prompt, input):
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        api_key =self.api_key ,
        # messages=[{"role":"system", "content":f"주어진 context에서 내용을 찾아 문서폼대로 작성해줘\n다음에 해당하는 내용이 없으면 '금일 기사에 관련 내용이 없습니다.'라고 내용에 적어줘\n\n문서폼:\n1.{company_name}의 주가(최근 종가 및 주가 변화)\n-\n2. {company_name}에 대한 투자의견(목표주가 등)\n-\n3.{company_name}의 사업 현황(신규계약 및 투자 관련)\n-\n4.{company_name}의 재무 현황(매출 및 영업이익 등)\n-\n5.{company_name}의  경영진 문제(사회 문제나 이슈)\n-\n6.{company_name}의  산업 현황 및 전망(관련 업종의 호황 및 불황)\n-"},{"role":"user", "content":f"context:{paragraph}"}],
        messages=[{"role":"system", "content":f"{prompt}"},{"role":"user", "content":f"content:{input}"}],
        # messages=[{"role":"system", "content":f"Find the contents in the given context and fill them out in the document titles.\n If there's nothing related to each part, just write, '기사에 관련된 내용이 없습니다.'\n\n document titles:\n1.{company_name}의 주가\n-content \n2.{company_name}에 대한 투자의견\n-content \n3.{company_name}의 사업 현황\n-content \n4.{company_name}의 재무 현황전망\n-content \n5.{company_name}의 경영진 문제\n-content \n6.{company_name}의 산업 현황 및 전망\n-content \n\nPlease follow the following guidelines for each title to be entered for each title.\n1.{company_name}의 주가: If there is any mention of actual stock trading price or closing price\n2. {company_name}에 대한 투자의견: If there is any comment on investment such as target stock price or buy/sell/neutral opinion\n3.{company_name}의 사업 현황: If you have any new contracts and investments, please fill them out.\n4.{company_name}의 재무 현황전망: If you have any sales and operating profit, please fill them out. Management issues in \n5.{company_name}의  경영진 문제: If there are any social issues or issues caused by management, or if there are any changes in management, please fill them out. Industry Status and Forecast in \n6.{company_name}의  산업 현황 및 전망: If you have any information about the boom and recession in company industry or related industries, please fill it out"},{"role":"user", "content":f"context:{paragraph}"}],
        temperature =0,
        max_tokens=1024,
        n=1)
        result_text = response["choices"][0]["message"]["content"]
        return result_text       
   
   
    def gpt_news_report(self, company_name, paragraph):
        if self.token_length(paragraph) > 3000:
            paragraph = paragraph[:2700]
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        api_key =self.api_key ,
        # messages=[{"role":"system", "content":f"주어진 context에서 내용을 찾아 문서폼대로 작성해줘\n다음에 해당하는 내용이 없으면 '금일 기사에 관련 내용이 없습니다.'라고 내용에 적어줘\n\n문서폼:\n1.{company_name}의 주가(최근 종가 및 주가 변화)\n-\n2. {company_name}에 대한 투자의견(목표주가 등)\n-\n3.{company_name}의 사업 현황(신규계약 및 투자 관련)\n-\n4.{company_name}의 재무 현황(매출 및 영업이익 등)\n-\n5.{company_name}의  경영진 문제(사회 문제나 이슈)\n-\n6.{company_name}의  산업 현황 및 전망(관련 업종의 호황 및 불황)\n-"},{"role":"user", "content":f"context:{paragraph}"}],
        messages=[{"role":"system", "content":f"Find the contents in the given context and fill them out in the document form.\n If there's nothing related to each part, just write, '기사에 관련된 내용이 없습니다.'.\n\n document form:\n1.{company_name}에 대한 투자의견(목표주가 등)\n-\n2.{company_name}의 주가(최근 종가 및 주가 변화)\n-\n3.{company_name}의 사업 현황(신규계약 및 투자 관련)\n-\n4.{company_name}의 재무 현황(매출 및 영업이익 등)\n-\n5.{company_name}의 경영진 문제(경영진의 문제나 이슈)\n-\n6.{company_name}의 산업 현황 및 전망(관련 업종의 호황 및 불황)\n-"},{"role":"user", "content":f"context:{paragraph}"}],
        # messages=[{"role":"system", "content":f"Find the contents in the given context and fill them out in the document titles.\n If there's nothing related to each part, just write, '기사에 관련된 내용이 없습니다.'\n\n document titles:\n1.{company_name}의 주가\n-content \n2.{company_name}에 대한 투자의견\n-content \n3.{company_name}의 사업 현황\n-content \n4.{company_name}의 재무 현황전망\n-content \n5.{company_name}의 경영진 문제\n-content \n6.{company_name}의 산업 현황 및 전망\n-content \n\nPlease follow the following guidelines for each title to be entered for each title.\n1.{company_name}의 주가: If there is any mention of actual stock trading price or closing price\n2. {company_name}에 대한 투자의견: If there is any comment on investment such as target stock price or buy/sell/neutral opinion\n3.{company_name}의 사업 현황: If you have any new contracts and investments, please fill them out.\n4.{company_name}의 재무 현황전망: If you have any sales and operating profit, please fill them out. Management issues in \n5.{company_name}의  경영진 문제: If there are any social issues or issues caused by management, or if there are any changes in management, please fill them out. Industry Status and Forecast in \n6.{company_name}의  산업 현황 및 전망: If you have any information about the boom and recession in company industry or related industries, please fill it out"},{"role":"user", "content":f"context:{paragraph}"}],
        temperature =0,
        max_tokens=1024,
        n=1)
        result_text = response["choices"][0]["message"]["content"]
        return result_text
    
    
    def summarize_mapreduce_chain(self, input_text, chunk_size, chunk_overlap):
        transcript = self.split_into_sentence(input_text)
        transcript = transcript.replace('\n', '\n\n')
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_text(transcript) # 문장 분리 길이 4000 미만 및 chunk_overlap 200
        docs = [Document(page_content=t) for t in texts]
        # 나뉜 문단별 토큰수 계산
        tmp_tok = 0
        for doc in docs:
            tokenized_text_chatgpt35 = self.enc35.encode(doc.page_content)
            # print(len(tokenized_text_chatgpt35))
            tmp_tok += len(tokenized_text_chatgpt35)
        # print('전체 토큰수', tmp_tok)
        # 템플릿 작성
        ############################################################################################################
        prompt_template = """다음 문장을 중요한 내용은 유지하면서 요약해주세요.:

        {text}

        한국어 문장요약 결과:
        """
        combine_prompt_template = """다음 문장들 중 중복된 내용을 제거하고 중요 내용은 유지하면서 요약해주세요.:

        {text}

        결과:
        """

        ############################################################################################################
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
        # 모든 데이터에 한번에 엑세스 하므로 전체 토큰이 
        llmc = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") # chatgpt 3.5 model
        chain = load_summarize_chain(llmc, 
                             chain_type="map_reduce",  # map_reduce 방식 각각의 chunk에 대해 요약을 먼저 진행하고 그 결과를 다시 요약하는 방식 -> 각각의 요약에대해 병렬처리함
                             return_intermediate_steps=True,
                             map_prompt=PROMPT,
                             combine_prompt=COMBINE_PROMPT,
                             verbose=False)
        res = chain({"input_documents": docs}, return_only_outputs=True)
        return res['output_text']
    
    def happycall_mapreduce_chain(self, input_text):
        transcript = self.split_into_sentence(input_text)
        transcript = transcript.replace('\n', '\n\n')
        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
        texts = text_splitter.split_text(transcript) # 문장 분리 길이 4000 미만 및 chunk_overlap 200
        docs = [Document(page_content=t) for t in texts]
        # 나뉜 문단별 토큰수 계산
        tmp_tok = 0
        for doc in docs:
            tokenized_text_chatgpt35 = self.enc35.encode(doc.page_content)
            # print(len(tokenized_text_chatgpt35))
            tmp_tok += len(tokenized_text_chatgpt35)
        # print('전체 토큰수', tmp_tok)
        # 템플릿 작성
        ############################################################################################################
        prompt_template = """다음 문장이 체크포인트를 준수하는지 확인하고 준수 여부와 해당되는 내용을 추출하여 아래 양식에 한국어 결과를 작성합니다

        체크포인트
        1. 본인정보 확인 여부: 고객의 이름, 전화번호, 주민등록번호 등을 확인하여 본인확인 절차를 진행하는 내용이 있는지
        2. 상품 정보 제공 여부: 상품의 기본항목(상품명, 가입기간, 조건 등)에 대해 설명하는 내용이 있는지
        3. 서류 정보 제공 여부: 약관 및 청약 서류와 같은 주요 문서에 대한 안내 및 제공 여부에 대한 내용이 있는지

        양식
        - 1. 본인정보 확인 여부: 있음(해당 부분 문장)/없음(내용이 없는 경우)
        - 2. 상품 정보 제공 여부: 있음(해당 부분 문장)/없음(내용이 없는 경우)
        - 3. 서류 정보 제공 여부: 있음(해당 부분 문장)/없음(내용이 없는 경우)
        
        문장: 
        {text}

        result:
        """
        combine_prompt_template = """다음 추출 결과를 취합하여 요약해주세요. 확인 사항이 '있음'인 경우 해당 부분 문장은 모두 합쳐서 보여주세요. 한번이라도 체크사항 기준에 대한 결과가 '있음'이 있으면 최종 결과는 '있음'입니다.
        요약 결과는 다음의 양식을 유지해주세요.
        
        양식
        - 1. 본인정보 확인 여부: 있음(해당 부분 문장)/없음(내용이 없는 경우)\n- 2. 상품 정보 제공 여부: 있음(해당 부분 문장)/없음(내용이 없는 경우)\n- 3. 서류 정보 제공 여부: 있음(해당 부분 문장)/없음(내용이 없는 경우)

        추출 결과:
        {text}

        요약 결과:
        """

        ############################################################################################################
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
        # 모든 데이터에 한번에 엑세스 하므로 전체 토큰이 
        llmc = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") # chatgpt 3.5 model
        chain = load_summarize_chain(llmc, 
                             chain_type="map_reduce",  # map_reduce 방식 각각의 chunk에 대해 요약을 먼저 진행하고 그 결과를 다시 요약하는 방식 -> 각각의 요약에대해 병렬처리함
                             return_intermediate_steps=True,
                             map_prompt=PROMPT,
                             combine_prompt=COMBINE_PROMPT,
                             verbose=False)
        res = chain({"input_documents": docs}, return_only_outputs=True)
        total_check = res['intermediate_steps']
        output = res['output_text']
        return total_check, output
       
    def faq_argumentation(self, text):
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        api_key =self.api_key,
        messages=[{"role":"system", "content":"You are a paraphrasing chatbot that creates fifteen sentences by changing the sentences you enter into various nouns and descriptive expressions with the same meaning. The meaning of the sentence should not be changed, new nouns should not be added, and only synonyms of previously used nouns can be used."},{"role":"user", "content":f"{text}"}],
        temperature =0.2,
        max_tokens=2848,
        n=1)
        result_text = response["choices"][0]["message"]["content"]
        sub_text = re.sub("[0-9]{1,2}\.","",result_text)
        result_list = sub_text.split("\n")
        results = list()
        for i in result_list:
            results.append(i.strip()) 
        return results    
        
    def kor_to_eng_translation(self, text):
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        api_key =self.api_key,
        messages=[{"role":"system", "content":"Your are chatbot that translates Korean input into English."},{"role":"user", "content":f"{text}"}],
        temperature =0,
        max_tokens=2848,
        n=2)
        result_text = response["choices"][0]["message"]["content"]
        return result_text    
  
    def chat_gpt_question_generation(self, pdf_file_path):
        quesiton_list = list()
        parsing_pages_text = self.ep.get_pdf_text(pdf_file_path)
        for page in parsing_pages_text:
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            api_key =self.api_key,
            messages=[{"role":"system", "content":"You are a chatbot that makes three pairs of Korean questions and answers in the given content. form is Q: A:"},{"role":"user", "content":f"content:{page}\n"}],
            temperature =0,
            max_tokens=1024,
            n=1)
            result_text = response["choices"][0]["message"]["content"]
            result_text = re.sub("[0-9]{1,2}\.","",result_text)
            splits = result_text.split("\n")
            quesiton_list.extend(splits)
        return quesiton_list
  
    def chat_gpt_qa_page(self, pdf_file_path, question):
        parsing_pages_text = self.ep.get_pdf_text(pdf_file_path)
        context = self.ep.openai_faiss_get_paragraph(question, parsing_pages_text)
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        api_key =self.api_key,
        messages=[{"role":"system", "content":"You are a chatbot that finds answer to the question in the context and what page is answer on.  You have to answer only the facts in the given context. If there is no answer in the context, please answer '해당 질문에대한 답을 찾을 수 없습니다.' "},{"role":"user", "content":f"context:{context}\nquestion:{question}"}],
        temperature =0,
        max_tokens=1024,
        n=1)
        result_text = response["choices"][0]["message"]["content"]
        return result_text
        
    def chat_gpt_qa(self, pdf_file_path, question):
        parsing_text = self.ps.parsing(pdf_file_path)
        paragraph_list, embedding_paragraph_list = self.ep.split_parsing_text(parsing_text,512)
        question_embedding = self.ep.get_question_embedding(question)
        embeddings = self.ep.doc_embdding_dict(embedding_paragraph_list)
        best_index_list = self.ep.get_best_paragraph_index(embeddings, question_embedding)
        paragraph_candidate = self.ep.sum_best_index_paragraphs(best_index_list, paragraph_list)
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        api_key =self.api_key,
        messages=[{"role":"system", "content":"You are a chatbot that finds answers to the question in the context. You have to answer only the facts in the given context. If there is no answer in the context, please answer '해당 질문에대한 답을 찾을 수 없습니다.'"},{"role":"user", "content":f"context:{paragraph_candidate}\nquestion:{question}"}],
        temperature =0,
        max_tokens=1024,
        n=1)
        result_text = response["choices"][0]["message"]["content"]
        return result_text
    
    def chat_gpt_summary(self, text, number):
        example_form = ""
        for i in range(number):
            example_form +=f"{i+1}."
            if i!= number-1:
                example_form +="\n" 
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        api_key =self.api_key,
        messages=[{"role":"system", "content":f"I am a chatbot that summarizes the given sentence. I make it in the form of the example form below.\n example form: {example_form}"},{"role":"user", "content":f"{text}"}],
        temperature =0.3,
        max_tokens=1500,
        n=1)
        result_text = response["choices"][0]["message"]["content"]
        return print(result_text)
    
    ################################################ 재미로 만든 것들##########################################################
    def gpt_juesus_counsel_answer(self, question):
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        api_key =self.api_key ,
        # messages=[{"role":"system", "content":f"주어진 context에서 내용을 찾아 문서폼대로 작성해줘\n다음에 해당하는 내용이 없으면 '금일 기사에 관련 내용이 없습니다.'라고 내용에 적어줘\n\n문서폼:\n1.{company_name}의 주가(최근 종가 및 주가 변화)\n-\n2. {company_name}에 대한 투자의견(목표주가 등)\n-\n3.{company_name}의 사업 현황(신규계약 및 투자 관련)\n-\n4.{company_name}의 재무 현황(매출 및 영업이익 등)\n-\n5.{company_name}의  경영진 문제(사회 문제나 이슈)\n-\n6.{company_name}의  산업 현황 및 전망(관련 업종의 호황 및 불황)\n-"},{"role":"user", "content":f"context:{paragraph}"}],
        messages=[{"role":"system", "content":f"You are a Jesus chatbot who consults the psychology of heartbroken people.\nI want you to use the Bible verse and Jesus' words to discuss the questions entered\nPlease answer in Korean only.\nPlease answer in the form given\n\nform:\n-도움이 될 성경구절:\n-답변:"},{"role":"user", "content":f"question:{question}"}],
        # messages=[{"role":"system", "content":f"Find the contents in the given context and fill them out in the document titles.\n If there's nothing related to each part, just write, '기사에 관련된 내용이 없습니다.'\n\n document titles:\n1.{company_name}의 주가\n-content \n2.{company_name}에 대한 투자의견\n-content \n3.{company_name}의 사업 현황\n-content \n4.{company_name}의 재무 현황전망\n-content \n5.{company_name}의 경영진 문제\n-content \n6.{company_name}의 산업 현황 및 전망\n-content \n\nPlease follow the following guidelines for each title to be entered for each title.\n1.{company_name}의 주가: If there is any mention of actual stock trading price or closing price\n2. {company_name}에 대한 투자의견: If there is any comment on investment such as target stock price or buy/sell/neutral opinion\n3.{company_name}의 사업 현황: If you have any new contracts and investments, please fill them out.\n4.{company_name}의 재무 현황전망: If you have any sales and operating profit, please fill them out. Management issues in \n5.{company_name}의  경영진 문제: If there are any social issues or issues caused by management, or if there are any changes in management, please fill them out. Industry Status and Forecast in \n6.{company_name}의  산업 현황 및 전망: If you have any information about the boom and recession in company industry or related industries, please fill it out"},{"role":"user", "content":f"context:{paragraph}"}],
        temperature =0.2,
        max_tokens=1524,
        n=1)
        result_text = response["choices"][0]["message"]["content"]
        return result_text
    
    def gpt_buddha_answer(self, question):
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        api_key =self.api_key ,
        # messages=[{"role":"system", "content":f"주어진 context에서 내용을 찾아 문서폼대로 작성해줘\n다음에 해당하는 내용이 없으면 '금일 기사에 관련 내용이 없습니다.'라고 내용에 적어줘\n\n문서폼:\n1.{company_name}의 주가(최근 종가 및 주가 변화)\n-\n2. {company_name}에 대한 투자의견(목표주가 등)\n-\n3.{company_name}의 사업 현황(신규계약 및 투자 관련)\n-\n4.{company_name}의 재무 현황(매출 및 영업이익 등)\n-\n5.{company_name}의  경영진 문제(사회 문제나 이슈)\n-\n6.{company_name}의  산업 현황 및 전망(관련 업종의 호황 및 불황)\n-"},{"role":"user", "content":f"context:{paragraph}"}],
        messages=[{"role":"system", "content":f"You are a chatbot that conveys Buddha's teachings.\nThe most relevant Buddhist to the question entered\nPlease answer using the wise saying\nAnd from the Buddha's point of view, please write a long answer to the question\nPlease answer in Korean"},{"role":"user", "content":f"question:{question}"}],
        # messages=[{"role":"system", "content":f"Find the contents in the given context and fill them out in the document titles.\n If there's nothing related to each part, just write, '기사에 관련된 내용이 없습니다.'\n\n document titles:\n1.{company_name}의 주가\n-content \n2.{company_name}에 대한 투자의견\n-content \n3.{company_name}의 사업 현황\n-content \n4.{company_name}의 재무 현황전망\n-content \n5.{company_name}의 경영진 문제\n-content \n6.{company_name}의 산업 현황 및 전망\n-content \n\nPlease follow the following guidelines for each title to be entered for each title.\n1.{company_name}의 주가: If there is any mention of actual stock trading price or closing price\n2. {company_name}에 대한 투자의견: If there is any comment on investment such as target stock price or buy/sell/neutral opinion\n3.{company_name}의 사업 현황: If you have any new contracts and investments, please fill them out.\n4.{company_name}의 재무 현황전망: If you have any sales and operating profit, please fill them out. Management issues in \n5.{company_name}의  경영진 문제: If there are any social issues or issues caused by management, or if there are any changes in management, please fill them out. Industry Status and Forecast in \n6.{company_name}의  산업 현황 및 전망: If you have any information about the boom and recession in company industry or related industries, please fill it out"},{"role":"user", "content":f"context:{paragraph}"}],
        temperature =0.2,
        max_tokens=1524,
        n=1)
        result_text = response["choices"][0]["message"]["content"]
        return result_text
     
    def DB_sales_gpt(self, clue_dict):
        #dict_sample={"last_visit_prodiuct":['암보험'],"age":"37세","sex":"남성","이름":"김일훈"}
        products = ",".join(clue_dict['last_visit_prodiuct'])
        name = clue_dict['이름']
        sex = clue_dict['sex']
        age = clue_dict['age'] 
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        api_key =self.api_key,
        messages = [{"role": "system", "content": "Please make a make a Korean statement recommending DB손해보험's insurance products based on the Customer Information\nIf there is Product information on the interesting product what among the Customer information, please make a phrase using the information\nPlease make a good statement so that you want to purchase an insurance product using the customer's age and gender\n\n Product information\n\nProduct Name:자동차보험\nProduct Features: Average compared to company offline\n\n17.7% cheaper, 37% off with special mileage discount\nProduct Name:암보험\nProduct Features: Additional diagnosis fee can be selected for each desired area\n\nProduct Name:어린이보험\nProduct Features:Guaranteed up to 100 years old\n\nProduct Name:운전자보험\nProduct Features:For the first time in the industry, police investigation fees can be guaranteed\n\n"},{"role": "user", "content": f"Customer Information:\n- interesting Product: {products}\n- name: {name}\n- sex: {sex}\n- age: {age}"}],
        temperature =0.3,
        max_tokens=1524,
        n=1)
        result_text = response["choices"][0]["message"]["content"]
        return result_text
    
    