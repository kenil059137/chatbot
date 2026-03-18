from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/pdf/student_support_policy_UGSF_PGSF.pdf")

docs = loader.load()

print("Pages:", len(docs))

for i, doc in enumerate(docs):
    print(f"\nPage {i+1} content preview:")
    print(doc.page_content[:200])
