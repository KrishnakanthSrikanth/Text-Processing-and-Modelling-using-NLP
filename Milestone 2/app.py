import os
import pickle
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
from flask import Flask, render_template, request, redirect
from bs4 import BeautifulSoup

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/accounting_finance')
def finance_accounting():
    return render_template('accounting_finance.html')


@app.route('/engineering')
def engineering():
    return render_template('engineering.html')


@app.route('/healthcare_nursing')
def healthcare_nursing():
    return render_template('healthcare_nursing.html')


@app.route('/sales')
def sales():
    return render_template('sales.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/<folder>/<filename>')
def article(folder, filename):
    return render_template('/' + folder + '/' + filename + '.html')


@app.route("/addJob", methods=['GET', 'POST'])
def addJob():
    if request.method == 'POST':

        # Read the .txt file
        f_title = request.form['title']
        f_content = request.form['description']
        # Classify the content
        if request.form['button'] == 'Classify':
            lang_mod = "lang_mod.pkl"
            with open(lang_mod, 'rb') as file:
                lang_model = pickle.load(file)

            # Same as in Milestone 1, we make the descriptions to lowercase, tokenize it and apply language model
            desc = f_content.lower()
            sentences = sent_tokenize(desc)
            pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
            tokenizer = RegexpTokenizer(pattern)
            token_lists = [tokenizer.tokenize(sen) for sen in sentences]
            tokenised_desc = list(chain.from_iterable(token_lists))

            # Using count vector to vectorize the data
            transformed_data = lang_model.transform([' '.join(article) for article in [tokenised_desc]])

            # Load the LR model
            pkl_filename = "logistic_model.pkl"
            with open(pkl_filename, 'rb') as file:
                model = pickle.load(file)

            # Predict the label of tokenized_data
            y_pred = model.predict(transformed_data)[0]
            y_pred = y_pred[0]

            # Since our target is 0,1,2,3 - we make them as necessary categories
            if int(y_pred) == 0:
                y_pred = 'accounting_finance'
            elif int(y_pred) == 1:
                y_pred = 'engineering'
            elif int(y_pred) == 2:
                y_pred = 'healthcare_nursing'
            else:
                y_pred = 'sales'

            return render_template('addJob.html', prediction=y_pred, title=f_title, description=f_content)

        elif request.form['button'] == 'Save':

            # First check if the recommended category is empty
            cat_recommend = request.form['category']
            if cat_recommend == '':
                return render_template('addJob.html', prediction=cat_recommend,
                                       title=f_title, description=f_content,
                                       category_flag='Recommended category must not be empty.')

            elif cat_recommend not in ['engineering', 'accounting_finance', 'healthcare_nursing', 'sales']:
                return render_template('addJob.html', prediction=cat_recommend,
                                       title=f_title, description=f_content,
                                       category_flag='Recommended category must belong to: engineering, accounting_finance, healthcare_nursing, sales.')

            else:

                # First read the html template
                soup = BeautifulSoup(open('templates/article_template.html'), 'html.parser')

                # Then adding the title and the content to the template
                # First, add the title
                div_page_title = soup.find('div', {'class': 'title'})
                title = soup.new_tag('h1', id='data-title')
                title.append(f_title)
                div_page_title.append(title)

                # Second, add the content
                div_page_content = soup.find('div', {'class': 'data-article'})
                content = soup.new_tag('p')
                content.append(f_content)
                div_page_content.append(content)

                # Finally write to a new html file
                filename_list = f_title.split()
                filename = '_'.join(filename_list)
                filename = cat_recommend + '/' + filename + ".html"
                with open("templates/" + filename, "w", encoding='utf-8') as file:
                    print(filename)
                    file.write(str(soup))

                # Clear the add-new-entry form and ask if user wants to continue to add new entry
                return redirect('/' + filename.replace('.html', ''))

    else:
        return render_template('addJob.html')


@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':

        if request.form['search'] == '':
            search_string = request.form["searchword"]

            # search over all the html files in templates to find the search_string
            article_search = []
            dir_path = 'templates'
            for folder in os.listdir(dir_path):
                if os.path.isdir(os.path.join(dir_path, folder)):
                    for filename in sorted(os.listdir(os.path.join(dir_path, folder))):
                        if filename.endswith('html'):
                            with open(os.path.join(dir_path, folder, filename), encoding="utf8") as file:
                                file_content = file.read()

                                # search for the string within the file
                                if search_string.lower() in file_content.lower():
                                    article_search.append([folder, filename.replace('.html', '')])

            # generate the right format for the Jquery script in search.html
            num_results = len(article_search)

            return render_template('search.html', num_results=num_results, search_string=search_string,
                                   article_search=article_search)

    else:
        return render_template('index.html')
