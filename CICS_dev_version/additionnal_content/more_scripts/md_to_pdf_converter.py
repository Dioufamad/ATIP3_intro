##### script to convert the readme written in markdown (.md) to pdf file for more plush distribution oriented files :
from markdown import markdown
import pdfkit

input_filename = '/home/diouf/ClassHD_work/ML/ClassHD/Full_documentation.md' # put here the name of the doc to convert # .md only
output_filename = '/home/diouf/ClassHD_work/ML/ClassHD/Full_documentation.pdf' # put here the name of the doc to obtain # .pdf only

with open(input_filename, 'r') as f:
    html_text = markdown(f.read(), output_format='html4')

pdfkit.from_string(html_text, output_filename)