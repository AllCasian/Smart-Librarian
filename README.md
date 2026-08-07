# 📚 Smart Librarian

Smart Librarian este un chatbot AI construit cu **Python**, **OpenAI SDK**, **ChromaDB** și **Streamlit**.

Aplicația recomandă cărți în funcție de interesele utilizatorului folosind tehnica **Retrieval-Augmented Generation (RAG)** și completează recomandarea prin **Tool Calling**, care returnează rezumatul complet al cărții recomandate.

Proiectul a fost realizat pentru assignment-ul **Essentials of LLM**.

---

# Funcționalități

## Funcționalități obligatorii

- ✅ 12 cărți stocate local în format JSON
- ✅ OpenAI Embeddings (`text-embedding-3-small`)
- ✅ ChromaDB ca vector database
- ✅ Semantic Search (RAG)
- ✅ Chatbot bazat pe OpenAI GPT
- ✅ Function Calling (`get_summary_by_title`)
- ✅ CLI
- ✅ Streamlit UI

## Funcționalități opționale

- ✅ Local Language Filter
- ✅ Image Generation folosind OpenAI Images API
- ✅ Conversation Memory
- ✅ Logging
- ✅ Unit Tests

---

# Tehnologii folosite

- Python 3.11+
- OpenAI SDK
- ChromaDB
- Streamlit
- python-dotenv
- pytest

---

# Arhitectură

```
                    User
                      │
                      ▼
            Language Filter
                      │
         (mesaj acceptat?)
          │                 │
          │ Nu              │ Da
          ▼                 ▼
 răspuns local       OpenAI Embedding
                            │
                            ▼
                       ChromaDB
                            │
                            ▼
                  Top 3 cărți relevante
                            │
                            ▼
                         GPT Model
                            │
                            ▼
               Tool Calling
        get_summary_by_title(title)
                            │
                            ▼
                Rezumat complet
                            │
                            ▼
                  Răspuns final
                            │
                            ▼
          (Opțional) Image Generation
                            │
                            ▼
                 Streamlit Interface
```

---

# Structura proiectului

```
smart-librarian/
│
├── app.py
├── streamlit_app.py
├── config.py
├── requirements.txt
├── README.md
├── .env.example
│
├── data/
│   ├── books.json
│   └── blocked_terms.txt
│
├── chroma_db/
│
├── scripts/
│   └── index_books.py
│
├── src/
│   ├── bootstrap.py
│   ├── book_repository.py
│   ├── conversation.py
│   ├── embeddings.py
│   ├── image_generation.py
│   ├── language_filter.py
│   ├── logging_config.py
│   ├── models.py
│   ├── recommendation_service.py
│   ├── retriever.py
│   ├── tool_definitions.py
│   ├── tools.py
│   └── vector_store.py
│
└── tests/
```

---

# Instalare

## 1. Clone repository

```bash
git clone <repository-url>

cd smart-librarian
```

---

## 2. Creează mediul virtual

Windows

```bash
python -m venv .venv
```

PowerShell

```bash
.venv\Scripts\Activate.ps1
```

CMD

```bash
.venv\Scripts\activate.bat
```

Linux / macOS

```bash
source .venv/bin/activate
```

---

## 3. Instalează dependențele

```bash
pip install -r requirements.txt
```

---

# Configurare

Creează fișierul:

```
.env
```

Pornind de la:

```
.env.example
```

Conținut:

```env
OPENAI_API_KEY=your_openai_key

OPENAI_CHAT_MODEL=gpt-4.1-mini

OPENAI_EMBEDDING_MODEL=text-embedding-3-small

OPENAI_IMAGE_MODEL=gpt-image-1
```

---

# Indexarea cărților

Înainte de prima rulare trebuie creat vector store-ul.

```bash
python -m scripts.index_books
```

Acest script:

- citește books.json
- generează embeddings
- creează ChromaDB
- salvează toate documentele

---

# Pornirea aplicației

## CLI

```bash
python app.py
```

---

## Streamlit

```bash
streamlit run streamlit_app.py
```

---

# Exemple de întrebări

```
Vreau o carte despre magie și prietenie.

Ce recomanzi pentru cineva interesat de război?

Vreau să citesc despre libertate și control social.

Ce este 1984?

Caut o poveste romantică.
```

---

# Cum funcționează RAG

1. Utilizatorul introduce întrebarea.

2. Întrebarea este transformată într-un embedding.

3. Embedding-ul este căutat în ChromaDB.

4. Sunt returnate cele mai relevante 3 cărți.

5. Acestea sunt trimise ca și context către GPT.

6. GPT selectează cea mai potrivită carte.

---

# Cum funcționează Tool Calling

După ce GPT selectează o carte:

```
The Hobbit
```

Modelul solicită apelarea funcției:

```
get_summary_by_title("The Hobbit")
```

Aplicația execută funcția Python și returnează:

- titlul
- autorul
- rezumatul complet

Rezultatul este trimis înapoi modelului, care generează răspunsul final.

---

# Cum funcționează filtrul de limbaj

Înainte ca mesajul să fie trimis către:

- OpenAI Embeddings
- ChromaDB
- GPT

acesta este verificat local.

Dacă sunt detectate cuvinte ofensatoare, aplicația răspunde politicos și oprește fluxul de procesare.

Astfel mesajele nepotrivite nu sunt trimise către modelul LLM.

---

# Generarea imaginilor

După recomandarea unei cărți, utilizatorul poate activa opțional generarea unei imagini.

Aplicația folosește OpenAI Images API pentru a genera o ilustrație originală inspirată de:

- titlul cărții
- autor
- rezumat

Imaginea este afișată în interfața Streamlit.

---

# Conversation Memory

Aplicația păstrează istoricul conversației pentru sesiunea curentă.

Acest lucru permite întrebări de tip:

```
Vreau o carte despre magie.

Ai ceva mai potrivit pentru adolescenți?
```

---

# Testare

Rulează testele:

```bash
pytest
```

sau

```bash
pytest -v
```

---

# Limitări

- Baza de date conține doar cărțile disponibile în `books.json`.
- Filtrul de limbaj este bazat pe o listă locală de termeni și nu detectează toate formele posibile de limbaj ofensator.
- Generarea imaginilor necesită un apel suplimentar către OpenAI Images API.
- Conversation Memory este păstrată doar pe durata sesiunii curente.

---

# Posibile îmbunătățiri

- Speech-to-Text
- Text-to-Speech
- React Frontend
- Docker
- Deploy în cloud
- Autentificare utilizatori
- Administrarea dinamică a cărților
- Moderation API pentru filtrarea limbajului
- Query rewriting pentru follow-up questions

---

# Autor

Proiect realizat folosind:

- Python
- OpenAI SDK
- ChromaDB
- Streamlit