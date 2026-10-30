import os
import streamlit as st
from google import genai
from google.genai import types
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 페이지 설정
st.set_page_config(
    page_title="2028 대입 개편 맞춤형 AI 플랫폼", page_icon="🎓", layout="centered"
)

# 2. Gemini API 키 설정 (Streamlit Secrets 또는 로컬)
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]

# 구글 공식 GenAI 클라이언트 초기화
client = genai.Client(api_key=api_key)

# 3. 상단 타이틀 및 탭 구성
st.title("🎓 2028 대입 개편 맞춤형 AI 플랫폼")
st.caption("스마트폰과 태블릿에서 설치 없이 바로 접속하여 이용하는 입시 도우미")

tab_chat, tab_calc, tab_dict = st.tabs(["💬 AI 입시 상담", "📊 5→9등급 환산", "📖 2028 대입 용어 단어장"])

# --- [탭 1: AI 입시 상담 (RAG 챗봇)] ---
with tab_chat:
    st.info('💡 서울대, 연세대, 고려대 2028학년도 입학전형 시행계획 PDF를 바탕으로 답변합니다. 문서에 없는 내용은 "제공된 입시 문서에 없는 정보입니다."라고 답변합니다.')

    # 캐시를 이용해 PDF 로드 및 ChromaDB 구축 (구글 공식 임베딩 사용)
    @st.cache_resource
    def load_vectorstore():
        pdf_files = ["서울대.pdf", "연세대.pdf", "고려대.pdf"]
        docs = []
        for file in pdf_files:
            if os.path.exists(file):
                loader = PyPDFLoader(file)
                docs.extend(loader.load())
        
        if not docs:
            return None
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # 인메모리 Chroma 클라이언트 생성
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(name="univ_2028")
        
        # 텍스트와 메타데이터 추출 후 구글 공식 API로 임베딩 생성 및 저장
        texts = [d.page_content for d in splits]
        metadatas = [d.metadata for d in splits]
        ids = [str(i) for i in range(len(texts))]
        
        # 배치 단위로 구글 임베딩 생성 (텍스트-임베딩-004 모델 사용)
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=batch_texts
            )
            embeddings = [e.values for e in response.embeddings]
            
            collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
                embeddings=embeddings
            )
            
        return collection

    collection = load_vectorstore()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("궁금한 입시 내용을 물어보세요 (예: 고려대 2028 전형 특징은?)"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            if collection is None:
                response_text = "⚠️ '서울대.pdf', '연세대.pdf', '고려대.pdf' 파일이 폴더에 없습니다. 파일을 업로드해 주세요!"
                st.markdown(response_text)
            else:
                with st.spinner("서·연·고 입시 문서를 분석하는 중..."):
                    # 1. 사용자 질문을 임베딩하여 유사한 문서 3개 검색
                    q_response = client.models.embed_content(
                        model="text-embedding-004",
                        contents=user_query
                    )
                    q_embedding = q_response.embeddings[0].values
                    
                    results = collection.query(
                        query_embeddings=[q_embedding],
                        n_results=3
                    )
                    
                    retrieved_docs = results['documents'][0] if results['documents'] else []
                    retrieved_metas = results['metadatas'][0] if results['metadatas'] else []
                    
                    context = "\n\n".join(retrieved_docs)
                    
                    # 2. Gemini 모델에 전달할 프롬프트 구성 (환각 방지 규칙 적용)
                    system_instruction = (
                        "당신은 2028학년도 대입 제도 개편안과 서·연·고 입시 전형을 완벽하게 이해하고 있는 입시 전문 AI입니다.\n"
                        "제공된 검색 문서 내용을 바탕으로 정확하게 답변하세요.\n"
                        "문서에 없는 내용은 절대로 지어내지 말고, 반드시 '제공된 입시 문서에 없는 정보입니다.'라고만 답하세요.\n\n"
                        "답변 하단에 반드시 어떤 대학의 몇 페이지를 참고했는지 '📄 참고 출처: 파일명 (페이지 X)' 형식으로 출처를 함께 적어주세요."
                    )
                    
                    prompt = f"참고 문서 내용:\n{context}\n\n사용자 질문: {user_query}"
                    
                    # 3. Gemini 답변 생성
                    chat_response = client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.0
                        )
                    )
                    response_text = chat_response.text
                    
                    # 4. 출처 정보 추가
                    if retrieved_metas:
                        sources = set()
                        for meta in retrieved_metas:
                            source_name = os.path.basename(meta.get("source", "PDF"))
                            page_no = meta.get("page", 0) + 1
                            sources.add(f"{source_name} ({page_no}페이지)")
                        if sources:
                            response_text += f"\n\n---\n📄 **참고 출처:** {', '.join(sources)}"

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

# --- [탭 2: 5등급 ➔ 9등급 환산 계산기] ---
with tab_calc:
    st.subheader("📊 5등급제 ➔ 9등급제 환산 계산기")
    st.write("2009년생부터 적용되는 새로운 고교 내신 5등급제 성적을 입력하면 기존 9등급제 기준 대략적인 위치를 환산해 드립니다.")
    
    grade_input = st.number_input("나의 5등급제 내신 등급 입력", min_value=1.0, max_value=5.0, step=0.1, value=1.5)
    if st.button("환산하기"):
        estimated_9 = (grade_input - 1) * (8 / 4) + 1
        st.success(f"입력하신 5등급제 **{grade_input}등급**은 기존 9등급제 환산 시 대략 **약 {estimated_9:.1f}등급** 수준에 해당합니다.")

# --- [탭 3: 2028 대입 핵심 용어 단어장] ---
with tab_dict:
    st.subheader("📖 2028 대입 핵심 용어 단어장")
    st.write("학생들이 자주 쓰지만 헷갈려하는 필수 입시 용어들을 정리했습니다.")
    
    search_term = st.text_input("🔍 단어 검색", placeholder="검색할 용어를 입력하세요...")
    
    terms = [
        ("수시모집", "학생부교과, 학생부종합, 논술 등 다양한 전형을 통해 9월경에 원서를 접수하고 대학에 지원하는 제도입니다."),
        ("정시모집", "11월에 치러지는 대학수학능력시험(수능) 성적을 중심으로 12월 말부터 원서를 접수하여 선발하는 전형입니다."),
        ("논술전형", "대학별로 출제하는 논술 고사 성적과 학생부(교과 등)를 반영하여 선발하는 수시 전형입니다."),
        ("학생부종합전형 (학종)", "학교생활기록부(교과 및 세특, 진로활동, 동아리 등)를 바탕으로 학생의 역량을 정성적으로 평가하여 뽑는 전형입니다."),
        ("수능 최저학력기준 (최저)", "수시 전형에서 최종 합격하기 위해 반드시 충족해야 하는 수능 성적의 최소 커트라인 조건입니다."),
        ("학생부교과전형 (교과)", "학교생활기록부의 교과 성적(내신)을 정량적으로 평가하여 주로 선발하는 수시 전형입니다."),
        ("고교 내신 5등급제", "2028 대입부터 적용되며, 기존 9등급제(상위 4% 1등급 등)가 상위 10% 1등급, 34% 2등급 등으로 개편되는 상대평가 제도입니다."),
        ("통합형 수능", "선택과목 없이 공통 과목 중심으로 수능 시험을 치러 수험생 간의 유불리를 줄이고 교육 정상화를 도모하는 체제입니다."),
        ("성취평가제 (진로/융합선택)", "점수가 아니라 학생의 성취도(A, B, C, D, E)를 평가하는 제도로, 2028 대입에서도 비중 있게 다뤄집니다."),
        ("입학전형시행계획", "대학교들이 대입 개편 방향에 맞춰 고등학교 2학년 때 미리 발표하는 해당 학년도 입시 가이드라인 공식 문서입니다.")
    ]
    
    for title, desc in terms:
        if not search_term or search_term.lower() in title.lower():
            with st.expander(f"📌 {title}"):
                st.write(desc)