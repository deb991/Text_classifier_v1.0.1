import pdfplumber as pdfreader


#pdf_file = "..\\res\\class.pdf"
pdf_file = "..\\res\\12578.pdf"


def read_file():
    with pdfreader.open(pdf_file) as file:
        pages = file.pages
        print('Pages:\t', pages)
        for page in pages:
            text = page.extract_text()
            print('Extracted Text:\t', text)
        return pages




read_file()