import os
import json
import io
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from neo4j import GraphDatabase

# 匯入文件處理套件
import PyPDF2
import docx

# ==========================================
# 1. 設定區 (Configuration)
# ==========================================
GOOGLE_API_KEY = "YOUR_API_KEY_HERE" # 替換為真實金鑰
NEO4J_URI = "YOUR_NEO4J_URI" # 替換為真實金鑰
NEO4J_USER = "YOUR_NEO4J_USER"  # 替換為真實金鑰
NEO4J_PASSWORD = "YOUR_NEO4J_PASSWORD" # 替換為真實金鑰

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

class GraphRequest(BaseModel):
    text: str

app = FastAPI(title="知識地圖自動建構平台 (雙引擎完整版)")

# ==========================================
# 2. 提示工程模板 (Prompt Engineering) - PPCD 模型
# ==========================================
SYSTEM_PROMPT = """
You are an expert Knowledge Graph Engineer and Product Development Specialist.
Your task is to extract structured knowledge following the PPCD (Person-Product-Concept-Document) model.

Entities (Nodes):
1. Person: id (Name), type (Person), role (Job Title), dept.
2. Product: id (Project Code/Model Name), type (Product), stage (EVT/DVT/PVT).
3. Concept: id (Key Term), type (Concept), category (Issue/Solution/Principle).
4. Document: id (Document Name/No), type (Document), stage (EVT/DVT/PVT), date.

Relationships (Edges):
- SAME_DEPT (Person -> Person)
- REPORTS_TO (Person -> Person)
- EVOLVES_FROM (Product -> Product)
- PART_OF (Product -> Product)
- SUPERSEDES (Product -> Product)
- SUB_CONCEPT_OF (Concept -> Concept)
- CONFLICTS_WITH (Concept -> Concept)
- PREREQUISITE_FOR (Concept -> Concept)
- RESPONSIBLE_FOR (Person -> Product)
- REVIEWS_AND_APPROVES (Person -> Product)
- INITIATES_REQUEST (Person -> Product)
- CONSULTS_ON (Person -> Product)
- MONITORS_PROGRESS (Person -> Product)
- PROPOSED_IDEA (Person -> Concept, must include: "evidence")
- SOLVED_ISSUE (Person -> Concept, must include: "evidence")
- DEFINES_STANDARD (Person -> Concept, must include: "evidence")
- MENTIONS_CONCEPT (Product -> Concept, must include: "evidence")
- HAS_TECHNICAL_ISSUE (Product -> Concept, must include: "evidence")
- IMPLEMENTS_SOLUTION (Product -> Concept, must include: "evidence")
- DEFINED_BY_SPEC (Product -> Concept, must include: "evidence")
- IDENTIFIES_RISK (Product -> Concept, must include: "evidence")
- REFERENCES (Document -> Document)
- VERSION_OF (Document -> Document)
- RECORDED_IN (Product -> Document)

CRITICAL INSTRUCTIONS:
1. Do NOT leave nodes isolated. You MUST infer explicit relationships from the context.
2. Role Weighting: For RESPONSIBLE_FOR, ONLY extract primary authors, decision-makers, or active engineers. IGNORE CC'd staff or purely administrative roles.
3. Hierarchy: If a product is a modified variant, derivative model, or sub-project of a main product family, use PART_OF. STRICT RULE: Do NOT extract physical parts, materials, or manufacturing tools (e.g., lens, film, jig) as Product nodes.
4. Traceability: If a product is recorded in a document, use RECORDED_IN. If a concept is described in detail, use ELABORATED_IN. If a document references another, use REFERENCES.
5. If the text explicitly mentions a management or reporting line between people, use REPORTS_TO.
6. LANGUAGE CONSTRAINT: You MUST extract and output all entity IDs, roles, and evidence in the EXACT ORIGINAL LANGUAGE of the input text (e.g., Traditional Chinese). Do NOT translate terms into English.
7. DOCUMENT CONSTRAINT: ONLY extract the exact filename explicitly provided at the beginning of the text in brackets (e.g., 【xxx.pdf】) as a Document node. Do NOT extract any general text, tasks, or action items (e.g., "相關治具評估") as Document nodes!

Output MUST be a valid JSON object. Strict JSON only. No markdown formatting.

Example Format:
{
  "nodes": [
    {"id": "Kevin Lin", "type": "Person", "role": "裁片工程師", "dept": "工程部"},
    {"id": "Angus", "type": "Person", "role": "組裝經理", "dept": "工程部"},
    {"id": "No.A 產品", "type": "Product", "stage": "EVT"},
    {"id": "鏡片曲度異常攤平", "type": "Concept", "category": "Issue"},
    {"id": "No.A 產品 EVT 測試異常報告.pdf", "type": "Document", "stage": "EVT"}
  ],
  "relationships": [
    {"source": "Kevin Lin", "target": "No.A 產品", "type": "RESPONSIBLE_FOR"},
    {"source": "Kevin Lin", "target": "Angus", "type": "REPORTS_TO"},
    {"source": "No.A 產品", "target": "鏡片曲度異常攤平", "type": "MENTIONS_CONCEPT", "evidence": "鏡片貼上偏光膜後發現曲度異常攤平"},
    {"source": "Kevin Lin", "target": "鏡片曲度異常攤平", "type": "PROPOSED_IDEA", "evidence": "Kevin Lin 提出此曲度異常分析與對策建議"},
    {"source": "No.A 產品", "target": "No.A 產品 EVT 測試異常報告.pdf", "type": "RECORDED_IN"}
  ]
}
"""

# ==========================================
# 3. 初始化 Gemini 模型與向量引擎
# ==========================================
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",  # 若 gemini-3-flash-preview 額度用盡，請務必先用 gemini-2.5-flash 或 gemini-2.5-flash-lite 測試 
    temperature=0.1,
    convert_system_message_to_human=True
)

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# ==========================================
# 3.5 混和推論演算法模組 (12條衍生關聯)
# ==========================================
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    return dot_product / (norm_v1 * norm_v2)

def infer_implicit_relationships(graph_data):
    print("🧠 啟動混合推論演算法：進行衍生關聯推論與相似度運算...")
    nodes = graph_data.get("nodes", [])
    rels = graph_data.get("relationships", [])
    new_relationships = []

    concepts = [n for n in nodes if n.get("type") == "Concept"]
    docs = [n for n in nodes if n.get("type") == "Document"]
    products = [n for n in nodes if n.get("type") == "Product"]
    persons = [n for n in nodes if n.get("type") == "Person"]

    # 1. 向量語意推論 (C-C): EQUIVALENT_TO, RELATED_TO
    if len(concepts) >= 2:
        concept_vectors = {}
        for c in concepts:
            try:
                concept_vectors[c["id"]] = embeddings_model.embed_query(c["id"])
            except Exception as e:
                print(f"⚠️ 向量化 [{c['id']}] 失敗: {e}")

        c_ids = list(concept_vectors.keys())
        for i in range(len(c_ids)):
            for j in range(i + 1, len(c_ids)):
                id1, id2 = c_ids[i], c_ids[j]
                sim = cosine_similarity(concept_vectors[id1], concept_vectors[id2])
                
                if sim >= 0.95:
                    new_relationships.append({"source": id1, "target": id2, "type": "EQUIVALENT_TO", "evidence": f"向量極度相似 ({sim:.2f})"})
                    print(f"   🔗 [等價映射] {id1} <--> {id2} (相似度: {sim:.2f})")
                elif 0.85 <= sim < 0.95:
                    new_relationships.append({"source": id1, "target": id2, "type": "RELATED_TO", "evidence": f"向量語意相關 ({sim:.2f})"})
                    print(f"   🔗 [語意相關] {id1} <--> {id2} (相似度: {sim:.2f})")

    # 2. 跨區索引錨定 (C-Do): ELABORATED_IN
    for doc in docs:
        for c in concepts:
            new_relationships.append({"source": c["id"], "target": doc["id"], "type": "ELABORATED_IN", "evidence": "推論自動文本索引"})

    # 3-1. EXPERTISE_IN (Pe-C)：頻次計數門檻
    expertise_counter = {}
    for r in rels:
        if r.get("type") in ["PROPOSED_IDEA", "SOLVED_ISSUE"]:
            src = r.get("source") or r.get("from")
            tgt = r.get("target") or r.get("to")
            if src and tgt:
                key = (src, tgt) 
                expertise_counter[key] = expertise_counter.get(key, 0) + 1

    EXPERTISE_THRESHOLD = 2 
    for (person_id, concept_id), count in expertise_counter.items():
        if count >= EXPERTISE_THRESHOLD:
            new_relationships.append({
                "source": person_id, "target": concept_id, "type": "EXPERTISE_IN", "evidence": f"高頻實戰戰功 (觸發次數: {count})"
            })
            print(f"   🎓 [專家認證] {person_id} 被標記為 [{concept_id}] 的領域專家！")

    # 3-2. COLLABORATES_WITH (Pe-Pe)
    product_responsibilities = {}
    for r in rels:
        if r.get("type") == "RESPONSIBLE_FOR":
            src = r.get("source") or r.get("from")
            tgt = r.get("target") or r.get("to")
            if src and tgt:
                product_responsibilities.setdefault(tgt, []).append(src)
    for p_id, person_list in product_responsibilities.items():
        for i in range(len(person_list)):
            for j in range(i+1, len(person_list)):
                new_relationships.append({"source": person_list[i], "target": person_list[j], "type": "COLLABORATES_WITH", "evidence": f"共同負責專案: {p_id}"})

    # 3-3. SHARES_PROCESS / DESIGN / COMPONENT (Pr-Pr)
    concept_mentions = {}
    for r in rels:
        if r.get("type") in ["MENTIONS_CONCEPT", "HAS_TECHNICAL_ISSUE"]:
            src = r.get("source") or r.get("from")
            tgt = r.get("target") or r.get("to")
            if src and tgt:
                concept_mentions.setdefault(tgt, []).append(src)
            
    concept_cat_map = {c["id"]: c.get("category", "") for c in concepts}
    for c_id, prod_list in concept_mentions.items():
        cat = concept_cat_map.get(c_id, "")
        rel_type = "SHARES_PROCESS" if cat == "Principle" else ("SHARES_DESIGN" if cat == "Solution" else "SHARES_COMPONENT")
        for i in range(len(prod_list)):
            for j in range(i+1, len(prod_list)):
                new_relationships.append({"source": prod_list[i], "target": prod_list[j], "type": rel_type, "evidence": f"共用技術節點: {c_id}"})

    # 4. 外部屬性對齊 (Mock Database)
    mock_hr_db = {"Kevin Lin": {"mentor": "Angus", "backup": "Dora"}}
    mock_sales_db = {"No.A產品": "Toyota", "No.B產品": "Toyota"}
    
    for p in persons:
        p_id = p["id"]
        if p_id in mock_hr_db:
            new_relationships.append({"source": mock_hr_db[p_id]["mentor"], "target": p_id, "type": "MENTOR_OF", "evidence": "HR 組織對照表映射"})
            new_relationships.append({"source": mock_hr_db[p_id]["backup"], "target": p_id, "type": "SUPPORTS_BACKUP", "evidence": "HR 組織對照表映射"})
            
    for i in range(len(products)):
        for j in range(i+1, len(products)):
            if mock_sales_db.get(products[i]["id"]) == mock_sales_db.get(products[j]["id"]):
                 new_relationships.append({"source": products[i]["id"], "target": products[j]["id"], "type": "SAME_CUSTOMER", "evidence": "業務專案客戶映射"})

    for i in range(len(docs)):
        for j in range(i+1, len(docs)):
            if docs[i]["id"][:4] == docs[j]["id"][:4]:
                new_relationships.append({"source": docs[i]["id"], "target": docs[j]["id"], "type": "SAME_PROJECT", "evidence": "專案代號字首比對"})

    # 5. 去重與合併回寫
    seen = set()
    for r in graph_data.get("relationships", []):
        seen.add(f"{r['source']}-{r['type']}-{r['target']}")
    
    for r in new_relationships:
        signature = f"{r['source']}-{r['type']}-{r['target']}"
        if signature not in seen:
            graph_data["relationships"].append(r)
            seen.add(signature)

    print(f"✅ 混合推論演算法執行完畢，共推導出 {len(new_relationships)} 條隱性與外部關聯。")
    return graph_data

# ==========================================
# 4. 核心大腦：文字轉圖譜處理邏輯
# ==========================================
def process_text_to_graph(text: str, source_filename: str = None):
    messages = [("system", SYSTEM_PROMPT), ("human", text)]
    response = llm.invoke(messages)
    
    raw_content = ""
    if isinstance(response.content, list):
        for part in response.content:
            if isinstance(part, str): raw_content += part
            elif isinstance(part, dict) and 'text' in part: raw_content += part['text']
    else:
        raw_content = str(response.content)
        
    json_str = raw_content.replace("```json", "").replace("```", "").strip()
    graph_data = json.loads(json_str)
    
    # ✅ 雙重保險與文件節點強制淨化機制
    if source_filename:
        # 1. 淨化：移除所有「不是真實檔名」的假文件節點
        graph_data["nodes"] = [
            n for n in graph_data.get("nodes", []) 
            if n.get("type") != "Document" or n.get("id") == source_filename
        ]
        
        # 2. 防呆：確保真實上傳檔案的節點絕對存在
        has_doc = any(n.get("type") == "Document" and n.get("id") == source_filename for n in graph_data["nodes"])
        if not has_doc:
            print(f"⚠️ 強制補建真實文件節點：{source_filename}")
            graph_data.setdefault("nodes", []).append({
                "id": source_filename,
                "type": "Document",
                "stage": "N/A",
                "date": "N/A"
            })
            
        # 3. 修補：若 LLM 產生了指向假文件的連線，強制將標靶導正回真實檔名
        for r in graph_data.get("relationships", []):
            if r.get("type") in ["RECORDED_IN", "ELABORATED_IN", "REFERENCES", "VERSION_OF"]:
                r["target"] = source_filename

        # 4. 強制錨定：將文本內的所有 Product 綁定至該檔案
        for n in graph_data.get("nodes", []):
            if n.get("type") == "Product":
                graph_data.setdefault("relationships", []).append({
                    "source": n["id"], "target": source_filename, "type": "RECORDED_IN", "evidence": "系統實體錨定"
                })

    # 啟動最新混合推論演算法 (這裡會自動將所有 Concept 建立 ELABORATED_IN 指向文件)
    graph_data = infer_implicit_relationships(graph_data)

    # 寫入 Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        for node in graph_data.get("nodes", []):
            node_id, node_type = node.get("id"), node.get("type") 
            if not node_id or not node_type: continue
            
            try:
                if node_type == "Person":
                    session.run("MERGE (p:Person {id: $id}) SET p.name=$id, p.role=$role, p.dept=$dept", id=node_id, role=node.get("role", ""), dept=node.get("dept", ""))
                elif node_type == "Product":
                    session.run("MERGE (p:Product {id: $id}) SET p.project_code=$id, p.stage=$stage", id=node_id, stage=node.get("stage", ""))
                elif node_type == "Concept":
                    session.run("MERGE (c:Concept {id: $id}) SET c.term=$id, c.category=$category", id=node_id, category=node.get("category", ""))
                elif node_type == "Document":
                    session.run("MERGE (d:Document {id: $id}) SET d.name=$id, d.stage=$stage, d.date=$date", id=node_id, stage=node.get("stage", ""), date=node.get("date", ""))
            except Exception as e:
                print(f"寫入節點錯誤: {e}")
        
        for rel in graph_data.get("relationships", []):
            source_id, target_id, rel_type = rel.get("source") or rel.get("from"), rel.get("target") or rel.get("to"), rel.get("type")
            if not all([source_id, target_id, rel_type]): continue
            
            try:
                ev = rel.get("evidence", "")
                query = f"MATCH (a {{id: $source_id}}) MATCH (b {{id: $target_id}}) MERGE (a)-[r:{rel_type}]->(b) SET r.evidence = $evidence"
                session.run(query, source_id=source_id, target_id=target_id, evidence=ev)
            except Exception as e:
                print(f"寫入關係錯誤: {e}")
                
    driver.close()
    return graph_data

# ==========================================
# 5. API 端點區
# ==========================================
@app.post("/api/build_graph")
def build_knowledge_graph(request: GraphRequest):
    try:
        # 純文字輸入沒有檔名參數
        graph_data = process_text_to_graph(request.text)
        return {"status": "success", "message": "文字圖譜建構完成", "extracted_data": graph_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload_and_build")
async def upload_and_build(file: UploadFile = File(...)):
    try:
        content = await file.read()
        extracted_text = ""
        if file.filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            extracted_text = "\n".join([p.text for p in doc.paragraphs])
        elif file.filename.endswith(".pdf"):
            pdf = PyPDF2.PdfReader(io.BytesIO(content))
            extracted_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        else:
            raise HTTPException(status_code=400, detail="僅支援 .pdf 或 .docx 格式")
            
        # ✅ 關鍵修改 1：將檔名強制加到文本最前方，引導 LLM 自然萃取
        extracted_text = f"【{file.filename}】\n\n" + extracted_text
            
        print(f"📄 成功從 {file.filename} 擷取文本，長度: {len(extracted_text)} 字")
        
        # ✅ 關鍵修改 2：將檔名傳遞給大腦，啟動強制補建防呆機制
        graph_data = process_text_to_graph(extracted_text, source_filename=file.filename)
        
        return {"status": "success", "message": f"檔案 {file.filename} 解析並建構完成", "extracted_data": graph_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"檔案處理失敗: {str(e)}")

@app.get("/api/get_global_graph")
def get_global_graph():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        nodes_map = {}
        rels_list = []
        with driver.session() as session:
            # 撈取 Neo4j 中所有實體節點（包含無連線之孤立節點/議題）
            result_nodes = session.run("MATCH (n) RETURN n")
            for record in result_nodes:
                node = record["n"]
                node_id = node.get("id")
                if node_id:
                    labels = list(node.labels)
                    node_type = labels[0] if labels else "Unknown"
                    
                    # ✅ 修正：依據 PPCD 模型，針對不同節點類型動態撈取專屬屬性
                    node_info = {
                        "id": node_id, 
                        "type": node_type
                    }
                    
                    if node_type == "Person":
                        node_info["role"] = node.get("role", "")
                        node_info["dept"] = node.get("dept", "")
                    elif node_type == "Product":
                        node_info["stage"] = node.get("stage", "")
                    elif node_type == "Concept":
                        node_info["category"] = node.get("category", "")
                    elif node_type == "Document":
                        node_info["stage"] = node.get("stage", "")
                        node_info["date"] = node.get("date", "")
                        
                    nodes_map[node_id] = node_info

            result_rels = session.run("MATCH (n)-[r]->(m) RETURN n.id AS source, m.id AS target, type(r) AS type, r.evidence AS evidence")
            for record in result_rels:
                rels_list.append({
                    "source": record["source"], "target": record["target"],
                    "type": record["type"], "evidence": record["evidence"] or ""
                })
        driver.close()
        return {"status": "success", "message": "全域知識庫載入完成", "extracted_data": {"nodes": list(nodes_map.values()), "relationships": rels_list}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 6. 網頁前端 (整合 PPCD、維度分色與雙引擎線型區分)
# ==========================================
@app.get("/", response_class=HTMLResponse)
def get_graph_ui():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>知識地圖自動建構平台 (雙引擎完整版)</title>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background-color: #f0f2f5;}
            .container { max-width: 1400px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
            h2 { color: #333; margin-bottom: 20px;}
            .input-section { display: flex; gap: 20px; margin-bottom: 20px; }
            .box { flex: 1; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }
            textarea { width: 100%; height: 80px; padding: 10px; border-radius: 5px; border: 1px solid #ccc; box-sizing: border-box; }
            button { width: 100%; padding: 12px; margin-top: 10px; background-color: #0056b3; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; font-weight: bold;}
            .global-btn { background-color: #17a2b8; width: 250px; margin-bottom: 20px; }
            #mynetwork { width: 100%; height: 800px; border: 1px solid #ccc; background-color: #ffffff; border-radius: 8px; margin-top: 15px;}
            
            /* 圖例面板優化 */
            .legend-panel { position: absolute; top: 15px; left: 15px; background: rgba(255,255,255,0.95); padding: 15px; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); z-index: 10; width: 260px; pointer-events: none;}
            .legend-title { margin: 0 0 10px 0; font-size: 14px; color: #2c3e50; font-weight: bold; border-bottom: 2px solid #eee; padding-bottom: 5px;}
            .legend-item { display: flex; align-items: flex-start; margin-bottom: 8px; font-size: 12px; color: #444; }
            .legend-color-box { width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; margin-top: 3px; flex-shrink: 0;}
            
            /* 實線與虛線的 CSS 繪製 */
            .line-solid { width: 20px; height: 3px; margin-right: 10px; margin-top: 7px; flex-shrink: 0; }
            .line-dashed { width: 20px; height: 0px; border-top: 3px dashed; margin-right: 10px; margin-top: 7px; flex-shrink: 0; }
            .legend-divider { border: 0; border-top: 1px dashed #ccc; margin: 10px 0; }

            /* ✅ 新增：調整游標懸停小視窗(Tooltip)的字體大小與樣式 */
            .vis-tooltip {
                font-size: 18px !important; 
                padding: 10px 15px !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
                line-height: 1.5 !important;

                /* 👇 新增這三行來控制自動換行 👇 */
                max-width: 400px !important;      /* 限制最大寬度 (400px 大約是 25 個中文字的寬度) */
                white-space: normal !important;   /* 允許文字自動換行 (覆蓋 Vis.js 原本的設定) */
                word-wrap: break-word !important; /* 確保中英文與長單字都能正確斷行 */
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🧠 知識地圖自動建構平台 (PPCD 雙引擎架構)</h2>
            <button class="global-btn" onclick="loadGlobalGraph()">🌐 載入全域知識庫 (Neo4j)</button>
            <div class="input-section">
                <div class="box">
                    <h3>📝 文字輸入</h3>
                    <textarea id="inputText" placeholder="貼上工程文件或會議紀錄..."></textarea>
                    <button onclick="buildGraphText()">✨ 生成圖譜</button>
                </div>
                <div class="box">
                    <h3>📂 檔案上傳</h3>
                    <input type="file" id="fileInput" accept=".pdf, .docx">
                    <button onclick="buildGraphFile()" style="background-color: #28a745;">🚀 上傳並生成</button>
                </div>
            </div>
            
            <!-- ✅ 新增：關鍵字搜尋區塊 -->
            <div style="margin-bottom: 15px; display: flex; gap: 10px;">
                <input type="text" id="searchInput" placeholder="🔍 輸入關鍵字搜尋節點 (例如：異常、Kevin、No.A)..." 
                       style="flex: 1; padding: 10px; border-radius: 5px; border: 1px solid #ccc; font-size: 14px;">
                <button onclick="searchNodes()" style="width: 120px; margin-top: 0; background-color: #6c757d;">搜尋節點</button>
                <button onclick="resetSearch()" style="width: 120px; margin-top: 0; background-color: #adb5bd; color: #333;">重置畫面</button>
            </div> 
            
            <p id="status">系統待命中...</p>
            
            <div style="position: relative;">
                <div id="mynetwork"></div>
                <div class="legend-panel" style="font-size: 13px;">
                    <h4 class="legend-title">🟢 實體節點</h4>
                    <div class="legend-item"><span class="legend-color-box" style="background: #E297FA; border: 1px solid #8A2BE2;"></span> <div><b>Product</b> (產品/專案)</div></div>
                    <div class="legend-item"><span class="legend-color-box" style="background: #D2E5FF; border: 1px solid #2B7CE9;"></span> <div><b>Person</b> (人員/工程師)</div></div>
                    <div class="legend-item"><span class="legend-color-box" style="background: #C2FABC; border: 1px solid #228B22;"></span> <div><b>Concept</b> (異常/原理/對策)</div></div>
                    <div class="legend-item"><span class="legend-color-box" style="background: #FFF59D; border: 1px solid #FBC02D;"></span> <div><b>Document</b> (文件/報告)</div></div>
                    
                    <div class="legend-divider"></div>
                    <h4 class="legend-title">🎨 關聯線段</h4>
                    <div class="legend-item"><div class="line-solid" style="background: #007BFF;"></div><div><b>Pe-Pe (人-人)</b><br><span style="color:#777; font-size:13px;">同僚、協作、代理等 (5種)</span></div></div>
                    <div class="legend-item"><div class="line-solid" style="background: #E83E8C;"></div><div><b>Pr-Pr (產品-產品)</b><br><span style="color:#777; font-size:13px;">共用、演進、取代等 (7種)</span></div></div>
                    <div class="legend-item"><div class="line-solid" style="background: #28A745;"></div><div><b>C-C (概念-概念)</b><br><span style="color:#777; font-size:13px;">等價、技術衝突/前提等 (5種)</span></div></div>
                    <div class="legend-item"><div class="line-solid" style="background: #DC3545;"></div><div><b>Pe-Pr (人-產品)</b><br><span style="color:#777; font-size:13px;">負責、審核、監控等 (5種)</span></div></div>
                    <div class="legend-item"><div class="line-solid" style="background: #FD7E14;"></div><div><b>Pe-C (人-概念)</b><br><span style="color:#777; font-size:13px;">具備專業、提議、解決等 (4種)</span></div></div>
                    <div class="legend-item"><div class="line-solid" style="background: #6F42C1;"></div><div><b>Pr-C (產品-概念)</b><br><span style="color:#777; font-size:13px;">技術異常、導入對策、識別風險等 (5種)</span></div></div>
                    <div class="legend-item"><div class="line-solid" style="background: #8D6E63;"></div><div><b>Document (跨區/文件)</b><br><span style="color:#777; font-size:13px;">記載於、同專案關聯等 (5種)</span></div></div>

                    <div class="legend-divider"></div>
                    <h4 class="legend-title">⚙️ 推論引擎</h4>
                    <div class="legend-item"><div class="line-solid" style="background: #555;"></div><div><b>LLM萃取</b><br><span style="color:#777; font-size:13px;">文本明示之直接關聯 (實線)</span></div></div>
                    <div class="legend-item"><div class="line-dashed" style="border-color: #555;"></div><div><b>混合推論</b><br><span style="color:#777; font-size:13px;">演算法推導之衍生關聯 (虛線)</span></div></div>

                    <div class="legend-divider"></div>
                    <p style="margin: 0; font-size: 13px; color: #666; line-height: 1.4;">
                        💡 <b>操作提示：</b><br>
                        • 游標停於線段可檢視確切關係名稱與佐證。<br>
                        • 節點大小依據拓撲權重自動縮放。<br>
                        • 點擊節點可高亮其關聯網絡。
                    </p>
                </div>
            </div>
        </div>

        <script>
            // ✅ 宣告全域變數，讓搜尋功能可以操作圖譜
            let network = null;
            let nodesDataSet = null;
            let edgesDataSet = null;

            async function buildGraphText() {
                const text = document.getElementById('inputText').value;
                if(!text) { alert("請輸入文字！"); return; }
                document.getElementById('status').innerText = "⏳ AI 正在解析文本，請稍候...";
                document.getElementById('status').style.color = "#f0ad4e";
                try {
                    const response = await fetch('/api/build_graph', {
                        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ text: text })
                    });
                    handleResponse(await response.json());
                } catch (error) { document.getElementById('status').innerText = "❌ 連線失敗：" + error; }
            }

            async function buildGraphFile() {
                const fileInput = document.getElementById('fileInput');
                if (fileInput.files.length === 0) { alert("請選擇檔案！"); return; }
                document.getElementById('status').innerText = "⏳ 正在讀取檔案並呼叫 AI 解析...";
                document.getElementById('status').style.color = "#f0ad4e";
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);
                try {
                    const response = await fetch('/api/upload_and_build', { method: 'POST', body: formData });
                    handleResponse(await response.json());
                } catch (error) { document.getElementById('status').innerText = "❌ 上傳失敗：" + error; }
            }

            async function loadGlobalGraph() {
                document.getElementById('status').innerText = "⏳ 正在從 Neo4j 載入全域知識庫...";
                document.getElementById('status').style.color = "#f0ad4e";
                try {
                    const response = await fetch('/api/get_global_graph');
                    handleResponse(await response.json());
                } catch (error) { document.getElementById('status').innerText = "❌ 讀取失敗：" + error; }
            }

            function handleResponse(result) {
                if (result.status === "success") {
                    document.getElementById('status').innerText = "✅ " + result.message;
                    document.getElementById('status').style.color = "#5cb85c";
                    drawGraph(result.extracted_data);
                } else {
                    document.getElementById('status').innerText = "❌ 發生錯誤：" + JSON.stringify(result);
                    document.getElementById('status').style.color = "#d9534f";
                }
            }

function drawGraph(data) {
                const safeNodes = data.nodes || [];
                const safeEdges = data.relationships || [];

                const degreeMap = {};
                safeEdges.forEach(edge => {
                    const fromId = edge.source || edge.from;
                    const toId = edge.target || edge.to;
                    degreeMap[fromId] = (degreeMap[fromId] || 0) + 1;
                    degreeMap[toId] = (degreeMap[toId] || 0) + 1;
                });

                // ✅ 修復：改用雙反斜線 + n (\\n) 產生正確的換行符號
                nodesDataSet = new vis.DataSet(safeNodes.map(n => {
                    let infoLines = [];
                    
                    if (n.id) infoLines.push(`id: "${n.id}"`);
                    if (n.type) infoLines.push(`type: "${n.type}"`);
                    
                    if (n.type === 'Product' && n.stage != null) infoLines.push(`stage: "${n.stage}"`);
                    if (n.type === 'Person' && n.role != null) infoLines.push(`role: "${n.role}"`);
                    if (n.type === 'Person' && n.dept != null) infoLines.push(`dept: "${n.dept}"`);
                    if (n.type === 'Concept' && n.category != null) infoLines.push(`category: "${n.category}"`);
                    if (n.type === 'Document' && n.stage != null) infoLines.push(`stage: "${n.stage}"`);
                    if (n.type === 'Document' && n.date != null) infoLines.push(`date: "${n.date}"`);

                    const customTitle = infoLines.join('\\n');

                    return {
                        id: n.id, 
                        label: n.id, 
                        group: n.type, 
                        title: customTitle, 
                        value: degreeMap[n.id] || 1,
                        hidden: false 
                    };
                }));

                // 將線段資料存入全域變數 edgesDataSet
                edgesDataSet = new vis.DataSet(safeEdges.map(r => {
                    const colorMap = {
                        'REPORTS_TO': '#007BFF', 'COLLABORATES_WITH': '#007BFF', 'SUPPORTS_BACKUP': '#007BFF', 'SAME_DEPT': '#007BFF', 'MENTOR_OF': '#007BFF',
                        'EVOLVES_FROM': '#E83E8C', 'PART_OF': '#E83E8C', 'SHARES_PROCESS': '#E83E8C', 'SHARES_DESIGN': '#E83E8C', 'SHARES_COMPONENT': '#E83E8C', 'SAME_CUSTOMER': '#E83E8C', 'SUPERSEDES': '#E83E8C',
                        'SUB_CONCEPT_OF': '#28A745', 'EQUIVALENT_TO': '#28A745', 'RELATED_TO': '#28A745', 'CONFLICTS_WITH': '#28A745', 'PREREQUISITE_FOR': '#28A745',
                        'RESPONSIBLE_FOR': '#DC3545', 'REVIEWS_AND_APPROVES': '#DC3545', 'INITIATES_REQUEST': '#DC3545', 'CONSULTS_ON': '#DC3545', 'MONITORS_PROGRESS': '#DC3545',
                        'PROPOSED_IDEA': '#FD7E14', 'SOLVED_ISSUE': '#FD7E14', 'EXPERTISE_IN': '#FD7E14', 'DEFINES_STANDARD': '#FD7E14',
                        'MENTIONS_CONCEPT': '#6F42C1', 'HAS_TECHNICAL_ISSUE': '#6F42C1', 'IMPLEMENTS_SOLUTION': '#6F42C1', 'DEFINED_BY_SPEC': '#6F42C1', 'IDENTIFIES_RISK': '#6F42C1',
                        'SAME_PROJECT': '#8D6E63', 'REFERENCES': '#8D6E63', 'VERSION_OF': '#8D6E63', 'RECORDED_IN': '#8D6E63', 'ELABORATED_IN': '#8D6E63'
                    };
                    
                    const edgeColor = colorMap[r.type] || '#cccccc';
                    
                    const secondOrderRels = ['EQUIVALENT_TO', 'RELATED_TO', 'ELABORATED_IN', 'EXPERTISE_IN', 'COLLABORATES_WITH', 'SHARES_PROCESS', 'SHARES_DESIGN', 'SHARES_COMPONENT', 'MENTOR_OF', 'SUPPORTS_BACKUP', 'SAME_CUSTOMER', 'SAME_PROJECT'];
                    const isSecondOrder = secondOrderRels.includes(r.type);

                    const bidirectionalRels = [
                        'COLLABORATES_WITH', 'SAME_DEPT', 'SHARES_PROCESS', 
                        'SHARES_DESIGN', 'SHARES_COMPONENT', 'SAME_CUSTOMER', 
                        'EQUIVALENT_TO', 'RELATED_TO', 'CONFLICTS_WITH', 'SAME_PROJECT'
                    ];
                    const arrowConfig = bidirectionalRels.includes(r.type) ? 'to, from' : 'to';

                    return {
                        from: r.source || r.from, 
                        to: r.target || r.to, 
                        label: r.type, 
                        font: { size: 16 }, 
                        arrows: arrowConfig,
                        color: { color: edgeColor, highlight: edgeColor, hover: edgeColor },
                        dashes: isSecondOrder ? [6, 6] : false, 
                        smooth: isSecondOrder ? false : { type: 'continuous' },
                        // ✅ 修復：將 <br> 改回雙反斜線 n (\\n)
                        title: "🏷️ 關係：" + r.type + "\\n⚙️ 引擎：" + (isSecondOrder ? "混合推論" : "LLM 萃取") + "\\n📄 佐證：" + (r.evidence || "系統自動推導"),
                        hidden: false
                    };
                }));

                const options = {
                    nodes: { shape: 'dot', scaling: { min: 15, max: 50 }, font: { face: 'Segoe UI', size: 35 } },
                    groups: {
                        Person: { color: { background: '#D2E5FF', border: '#2B7CE9', hover: { background: '#D2E5FF', border: '#2B7CE9' }, highlight: { background: '#D2E5FF', border: '#2B7CE9' } } },
                        Product: { color: { background: '#E297FA', border: '#8A2BE2', hover: { background: '#E297FA', border: '#8A2BE2' }, highlight: { background: '#E297FA', border: '#8A2BE2' } } },
                        Concept: { color: { background: '#C2FABC', border: '#228B22', hover: { background: '#C2FABC', border: '#228B22' }, highlight: { background: '#C2FABC', border: '#228B22' } } },
                        Document: { color: { background: '#FFF59D', border: '#FBC02D', hover: { background: '#FFF59D', border: '#FBC02D' }, highlight: { background: '#FFF59D', border: '#FBC02D' } } }
                    },
                    edges: { font: { align: 'top', size: 10, color: 'rgba(0,0,0,0)' }, smooth: { type: 'continuous' } },
                    interaction: { zoomSpeed: 0.1, hover: true, tooltipDelay: 200 },
                    physics: { 
                        barnesHut: { 
                            gravitationalConstant: -10000, 
                            centralGravity: 0.05,          
                            springLength: 350,             
                            springConstant: 0.015,         
                            damping: 0.25,                 
                            avoidOverlap: 1                
                        },
                        stabilization: {
                            iterations: 150,               
                            fit: true                      
                        }
                    }
                };
                
                const container = document.getElementById('mynetwork');
                network = new vis.Network(container, { nodes: nodesDataSet, edges: edgesDataSet }, options);
            }

            // ✅ 實作搜尋節點功能
            function searchNodes() {
                if (!network || !nodesDataSet) return;
                const keyword = document.getElementById('searchInput').value.toLowerCase();
                
                if (!keyword) {
                    resetSearch();
                    return;
                }

                const allNodes = nodesDataSet.get();
                const matchedNodes = [];

                // 搜尋比對
                allNodes.forEach(node => {
                    const match = node.id.toLowerCase().includes(keyword) || (node.title && node.title.toLowerCase().includes(keyword));
                    if (match) {
                        matchedNodes.push(node.id);
                        nodesDataSet.update({ id: node.id, hidden: false, font: {size: 20, color: '#ff0000', bold: true} }); // 高亮放大
                    } else {
                        nodesDataSet.update({ id: node.id, color: { background: 'rgba(200,200,200,0.3)', border: 'rgba(150,150,150,0.3)' }, font: {color: 'rgba(0,0,0,0.2)'} });
                    }
                });

                // 淡化無關連線
                const allEdges = edgesDataSet.get();
                allEdges.forEach(edge => {
                    if (matchedNodes.includes(edge.from) || matchedNodes.includes(edge.to)) {
                        edgesDataSet.update({ id: edge.id, hidden: false });
                    } else {
                        edgesDataSet.update({ id: edge.id, color: { color: 'rgba(200,200,200,0.1)' } });
                    }
                });

                if (matchedNodes.length > 0) {
                    network.fit({ nodes: matchedNodes, animation: true }); // 自動縮放
                    document.getElementById('status').innerText = `✅ 找到 ${matchedNodes.length} 個符合的節點`;
                } else {
                    document.getElementById('status').innerText = `⚠️ 找不到符合「${keyword}」的節點`;
                }
            }

            // ✅ 實作還原畫面功能
            function resetSearch() {
                if (!nodesDataSet) return;
                document.getElementById('searchInput').value = "";
                document.getElementById('status').innerText = "✅ 畫面已還原";
                
                if (document.getElementById('status').innerText.includes("還原")) {
                    loadGlobalGraph(); // 直接重新載入全域圖譜來恢復所有顏色
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content

# ==========================================
# 7. 本地端伺服器啟動區塊
# ==========================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 正在啟動 PPCD 雙引擎知識圖譜伺服器...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
