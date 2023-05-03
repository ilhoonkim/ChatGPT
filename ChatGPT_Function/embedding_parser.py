import re
import os, sys
from transformers import AutoTokenizer, AutoModel
from pdf_parser import Parser
import torch
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer, util
import numpy as np
from copy import deepcopy
import pdfplumber
import openai
import faiss 


openai.api_key = "chatgpt api key"

def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)

class embedding_parser:
    def __init__(self):
        # self.embedding_model = AutoModel.from_pretrained('jhgan/ko-sroberta-multitask')
        # self.embedding_tokenizer = AutoTokenizer.from_pretrained('jhgan/ko-sroberta-multitask')
        self.embedding_model = AutoModel.from_pretrained('bongsoo/moco-sentencebertV2.0')
        self.embedding_tokenizer = AutoTokenizer.from_pretrained('bongsoo/moco-sentencebertV2.0')



    def openai_faiss_get_paragraph(self, question, page_tedx_list):
        xq = array(openai.Embedding.create(input=question, engine="text-embedding-ada-002")['data'][0]['embedding']).reshape(1,1536)
        index = faiss.IndexFlatL2(1536)
        embedding_list = list()
        for i in page_tedx_list:
            # set end position of batch
            embeds = openai.Embedding.create(input=i, engine="text-embedding-ada-002")['data'][0]['embedding']
            # prep metadata and upsert batch
            embedding_list.append(embeds)
        embedding_array = array(embedding_list)
        index.add(embedding_array)
        distance, indices = index.search(xq, 1)
        result = page_tedx_list[indices.tolist()[0][0]]
        return result

    def get_pdf_text(self, pdf_path):
        page_content = dict()
        with pdfplumber.open(pdf_path) as pdf:
            # 페이지를 반복하며 텍스트와 테이블 데이터 추출
            for idx, page in enumerate(pdf.pages):
                text = page.extract_text()
                tables = page.extract_tables()
                page_content[str(idx+1)] = str(text) + str(tables)

        page_contet_list = list()
        for k,v in page_content.items():
            page_contet_list.append(f"{k}페이지 내용: {v}")
        return page_contet_list
 
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def doc_embdding_dict(self, embedding_paragraph_list):
        paragraphs = list()
        for p in embedding_paragraph_list:
            paragraphs.append(p[1])
        encoded_input = self.embedding_tokenizer(paragraphs, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask']).numpy()
        embeddings_list = list()
        for idx, p in enumerate(embedding_paragraph_list):
            embeddings_list.append((p[0],embeddings[idx]))
        return embeddings_list
    
    def doc_page_embdding_dict(self, embedding_paragraph_list):
        paragraphs = list()
        for p in embedding_paragraph_list:
            paragraphs.append(p)
        encoded_input = self.embedding_tokenizer(paragraphs, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask']).numpy()
        embeddings_list = list()
        for idx, p in enumerate(embedding_paragraph_list):
            embeddings_list.append((idx,embeddings[idx]))
        return embeddings_list
    
    def split_doc_text_length(self, doc_text, max_seq_length):
        temp_len = 0
        paragraphs_list = list()
        paragraphs = doc_text.split("\\n")
        temp_list = list()
        for idx, paragraph in enumerate(paragraphs):
            if idx == len(paragraphs)-1:
                paragraphs_list.append("\\n".join(temp_list))
            else:
                temp_len+=len(self.embedding_tokenizer(paragraph)['input_ids'])
                if temp_len < max_seq_length:
                    temp_list.append(paragraph)
                else:
                    paragraphs_list.append("\\n".join(temp_list))
                    temp_len = 0
                    temp_list = list()
                    temp_list.append(paragraph)
        return paragraphs_list

    def get_question_embedding(self, question):
        encoded_input = self.embedding_tokenizer(question, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
        question_embedding = self.mean_pooling(model_output, encoded_input['attention_mask']).numpy()
        return question_embedding


    # def get_best_paragraph_index(self, paragraphs_embeddings, qeustion_embeddings):
    #     doc_embeddings = list()
    #     for pe in paragraphs_embeddings:
    #         doc_embeddings.append(pe[1])
    #     cosine_scores = util.dot_score(qeustion_embeddings, doc_embeddings)
    #     scores = list(cosine_scores[0].numpy())
    #     best_index = scores.index(max(scores))
    #     para_best_index = paragraphs_embeddings[best_index][0]
    #     return para_best_index

    def get_best_paragraph_index(self, paragraphs_embeddings, qeustion_embeddings):
        doc_embeddings = list()
        for pe in paragraphs_embeddings:
            doc_embeddings.append(pe[1])
        cosine_scores = util.dot_score(qeustion_embeddings, doc_embeddings)
        scores = list(cosine_scores[0].numpy())
        scores_sorted = deepcopy(scores)
        scores_sorted.sort(reverse=True)
        best_index = list()
        for score in scores_sorted[:1]:
            best_index.append(scores.index(score))
        para_best_index_list = list()
        for b in best_index:
            para_index = paragraphs_embeddings[b][0]
            para_best_index_list.append(para_index)
        return para_best_index_list

    def sum_best_index_paragraphs(self, best_index_list, paragraph_list):
        candidates_sum =""
        for idx, bi in enumerate(best_index_list):
            candidates = paragraph_list[bi]
            if len(self.embedding_tokenizer(candidates)['input_ids']) < 1500:
                candidates_sum =candidates_sum +"\\n"+candidates
            else:
                if idx ==0:
                    candidates_sum = candidates
                else:
                    if len(self.embedding_tokenizer(candidates_sum)['input_ids']) < 1500:
                        candidates_sum =candidates_sum +"\\n"+candidates
                    else:
                        break
        return candidates_sum

    def split_parsing_text(self, parsing_text, token_num):
        doc_text = parsing_text.split("\\n")
        paragraph_list = list()
        for p in doc_text:
            if re.search("[0-9]{1,2}\.|\([0-9]{1,2}\)|\<[0-9]{1,2}\>|[0-9]{1,2}\)|①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|⑪|⑫|⑬|⑭|⑮|⑯|\【.{3,18}\】", p):
                paragraph_list.append(p)
            else:
                sum_para = paragraph_list[-1] + p
                paragraph_list.pop(-1)
                paragraph_list.append(sum_para)    

        embedding_paragraph_list = list()
        for idx, pp in enumerate(paragraph_list):
            if len(self.embedding_tokenizer(pp)['input_ids']) <= token_num:
                embedding_paragraph_list.append((idx,pp))                 
            elif pp.count("}") > 1 and pp.count("{"):
                rpp = pp.replace("}","}●")
                splits = rpp.split("●")
                sum = 0
                sum_para = ""           
                for rp in splits:
                    if sum < token_num:
                        sum+=len(self.embedding_tokenizer(rp)['input_ids'])
                        sum_para +=f"\\n{rp}"
                    else:
                        embedding_paragraph_list.append((idx,sum_para))
                        sum = 0
                        sum_para = ""
                        sum+=len(self.embedding_tokenizer(rp)['input_ids'])
            else:
                embedding_paragraph_list.append((idx,pp))
                
        return paragraph_list, embedding_paragraph_list
