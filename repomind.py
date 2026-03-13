import os
import json
import argparse
import subprocess
import shutil
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

EXTENSION_TO_LANGUAGE = {
    ".py"  : Language.PYTHON,
    ".js"  : Language.JS,
    ".ts"  : Language.JS,
    ".jsx" : Language.JS,
    ".tsx" : Language.JS,
    ".java": Language.JAVA,
    ".go"  : Language.GO,
    ".rs"  : Language.RUST,
    ".cpp" : Language.CPP,
    ".c"   : Language.C,
    ".cs"  : Language.CSHARP,
    ".rb"  : Language.RUBY,
    ".md"  : Language.MARKDOWN,
    ".rst" : Language.RST,
    ".html": Language.HTML,
}

TEXT_EXTENSIONS = {
    ".txt", ".json", ".yaml", ".yml",
    ".toml", ".sql", ".sh", ".csv",
    ".xml", ".cfg", ".ini"
}

IGNORE_DIRS = {"__pycache__", ".git", "node_modules", "venv", ".env"}

def is_github_url(repo):

    if repo.startswith("https://github.com") or repo.startswith("git@github.com"):
        return True
    return False
   

def clone_repo(github_url, target_dir="./cloned_repo"):

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    print(f"Cloning the Git Repo from {github_url}")

    result= subprocess.run(
        ["git","clone","--depth=1", github_url, target_dir],
        capture_output=True,
        text=True
    )
    if result.returncode!=0:
        raise RuntimeError(f"Clone failed: {result.stderr}")

    print(f"Cloned to {target_dir}")
    return target_dir
    

def cleanup_repo(target_dir):
    print(f" Deleting the repo{target_dir}")
    shutil.rmtree(target_dir)

def get_splitter(extension):
   
    if extension in EXTENSION_TO_LANGUAGE:
        splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=512,
            chunk_overlap=0,
            language=EXTENSION_TO_LANGUAGE[extension]
        )
        return splitter
    else:
        # fallback for yaml, json, sql, toml, txt etc
        return RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=50
        )
    
def get_line_numbers(full_text, chunk):

    # fins the index of where the chunk starts. This gives the character index
    index=full_text.find(chunk)
    if index==-1:
        return (0,0)
    # we are counting number of lines from start to the index which gives starting lines of the chunk
    start_pos= full_text[:index].count("\n")+1
    # to get the end line of the chunk we count number of lines present in the chunk and add it to the starting line number which we found
    end_pos= start_pos+chunk.count("\n")

    return (start_pos,end_pos)


def load_codebase(repo_path):

    documents=[]
    lang_count = Counter()
    total_chunks=0
    file_count = 0
    
    for root,dirs,files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            _,extension= os.path.splitext(file)
            
            if extension not in EXTENSION_TO_LANGUAGE and extension not in TEXT_EXTENSIONS:
                continue
            file_count+=1
            filepath= os.path.join(root,file)
            lang_count[extension] += 1
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
            except (UnicodeDecodeError, PermissionError):
                print(f" Skipping unreadable file: {filepath}")
                continue
            splitter= get_splitter(extension)
            chunks=splitter.split_text(text)
            total_chunks+=len(chunks)
            for chunk in chunks:
                start_line, end_line= get_line_numbers(text,chunk)
                doc= Document(
                    page_content=chunk,
                    metadata={
                        "source":os.path.relpath(filepath,repo_path),
                        "language":extension,
                        "lines":f"lines {start_line} - {end_line}"
                    }
                )
                documents.append(doc)
   
    print(f"Files loaded: {file_count}")
    print(f"Total chunks: {total_chunks}")
    print(f"languages: {dict(lang_count)}")
    return documents
def build_or_load_vectorstore(docs, persist_dir):
    embeddings = OpenAIEmbeddings(model="openai/text-embedding-3-large")

    vector_data=Chroma(
        embedding_function=embeddings,
        persist_directory= persist_dir
    )
    if vector_data._collection.count()<1:
        print(f"Embedding {len(docs)} chunks")
        vector_data=Chroma.from_documents(
            docs,
            embeddings,
            persist_directory= persist_dir
        )    
        print("Stored")
    else:
        print(f"Loaded existing Vector DB ")
    return vector_data
def search(query, vectorstore, n_results=3):

    related_chunks= vectorstore.similarity_search(query,k=n_results)
    return related_chunks

def generate_answer(query, chunks, llm):
    
    context=""
    for chunk in chunks:
        context+=chunk.page_content+"\n\n"
    prompt=f"""
    You are an expert code assistant helping a developer understand a codebase.
    Given the following code snippets, answer the question clearly.
    - Mention the specific file and function where relevant
    - Be concise — developers want direct answers
    - If the code doesn't contain the answer, say so clearly

    Code context:
    {context}

    Question: {query}
    """
    response= llm.invoke(prompt)
    return response.content


def evaluate(query, answer, top_chunk, judge):

    def score_answer_relevance(query,answer):
        prompt=f"""
        On a scale of 0 to 1 how well is the answer addressing the query?
        question:{query}
        answer:{answer}
        Reply only with a JSON :{{"score":0 to 1, "reason":"Reason for the given score"}}
        """
        response = judge.invoke(prompt)
        content = response.content.strip().strip("```json").strip("```").strip()
        data = json.loads(content)
        return data["score"], data["reason"]
    def score_context_relevance(query, top_chunk):
        prompt=f""" 
        On a scale of 0 to 1, how relevant is this chunk to answering the query?
        Query: {query}
        Chunk: {top_chunk}
        Reply ONLY with a JSON: {{"score": 0.0, "reason": "..."}}
        
        """
        response = judge.invoke(prompt)
        content = response.content.strip().strip("```json").strip("```").strip()
        data = json.loads(content)
        return data["score"], data["reason"]
    

    def score_faithfulness(answer, top_chunk):

        prompt = f""" 
        On a scale of 0 to 1, how faithful is this answer to the context?
        (1 = only uses context, 0 = contains hallucinated facts)
        Context: {top_chunk}
        Answer: {answer}
        Reply ONLY with a JSON: {{"score": 0.0, "reason": "..."}}
        """
        response= judge.invoke(prompt)
        content = response.content.strip().strip("```json").strip("```").strip()
        data = json.loads(content)
        return data["score"], data["reason"]
    
    cr_score,reason_cr= score_context_relevance(query,top_chunk)
    ar_score,reason_ar=score_answer_relevance(query,answer)
    fa_score,reason_fa=score_faithfulness(answer,top_chunk)
    return {
    "context_relevance": {"score": cr_score, "reason": reason_cr},
    "faithfulness":      {"score": fa_score, "reason": reason_fa},
    "answer_relevance":  {"score": ar_score, "reason": reason_ar}
    }

def main():
    
    parser=argparse.ArgumentParser(description="Codebase Q&A using RAG")
    parser.add_argument("--repo",  type=str, required=True,
                        help="Local path OR GitHub URL")
    parser.add_argument("--query", type=str, required=True,
                        help="Question about the codebase")
    args=parser.parse_args()
    repo=args.repo
    query=args.query
    llm=ChatOpenAI(
        model="openai/gpt-3.5-turbo"
    )
    judge=ChatOpenAI(
        model="openai/gpt-3.5-turbo"
    )

    if is_github_url(repo):
        repo_name = repo.split("/")[-1]
    else:
        repo_name = os.path.basename(os.path.abspath(repo))
    
    persist_dir = f"./repomind_db_{repo_name}" 

    if os.path.exists(persist_dir):
        print(f"Cache found for '{repo_name}'. Skipping clone + embed.")
        docs = []   # not needed, DB already has everything
    else:
        # Only clone + load if no cache exists
        cloned      = False
        if is_github_url(repo):
            cloned_path = clone_repo(repo)
            repo_path   = cloned_path
            cloned      = True
        else:
            repo_path = repo
            

        print("Loading codebase..")
        docs= load_codebase(repo_path)
        # Use repo name as the persist_dir
        
        if cloned:
            cleanup_repo(cloned_path)
    vector_data=build_or_load_vectorstore(docs,persist_dir)
    related_chunks=search(query,vector_data)
    answer= generate_answer(query,related_chunks,llm)
    top_chunk=related_chunks[0]
    
    print(f"The answer is :{answer}")
    print(f"The top source is :{top_chunk.metadata['source']}")
    scores = evaluate(query, answer, top_chunk.page_content, judge)
    print(f"  Context Relevance : {scores['context_relevance']['score']} — {scores['context_relevance']['reason']}")
    print(f"  Faithfulness      : {scores['faithfulness']['score']} — {scores['faithfulness']['reason']}")
    print(f"  Answer Relevance  : {scores['answer_relevance']['score']} — {scores['answer_relevance']['reason']}")


if __name__ == "__main__":
    main()




