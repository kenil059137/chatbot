from langchain_community.document_loaders import SeleniumURLLoader

urls = ["https://www.charusat.ac.in/course-single"]

loader = SeleniumURLLoader(urls=urls)
docs = loader.load()

print(f"Total docs: {len(docs)}")
print(f"Content length: {len(docs[0].page_content)}")
print("\n--- Preview ---")
print(docs[0].page_content[:500])