# flask -> framework for web app 
# request -> allow access to incoming request data 
# jsonify -> to convert python dictionaries to JSON format
# render_template -> renders HTML templates
# answer_question -> custom function responsible for answering questions using transformers 
from flask import Flask, request, jsonify, render_template
from answer_question import answer_question
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# create instance of Flask app 
app = Flask(__name__)

# Default model name = name of default transformer model to be used in case no specific model specified 
DEFAULT_MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"

# Load the default model and tokenizer
# default transformer model and tokenizer loaded during app startup 
app.last_used_model_name = DEFAULT_MODEL_NAME
app.model = AutoModelForQuestionAnswering.from_pretrained(app.last_used_model_name)
app.tokenizer = AutoTokenizer.from_pretrained(app.last_used_model_name)

# route defined to handle both GET & POST requests 
@app.route('/', methods=['GET', 'POST'])
def get_answer():
    # for POST requests, it processes the form data submitted by user 
    # check if http request method is POST, if true means form has been submitted with data
    if request.method == 'POST':
        # extract the form data submitted by user {question, reference, model_name}
        # Parse input
        # This line extracts the form data submitted with the POST request. 
        # In Flask, request.form is a dictionary-like object containing the key-value pairs of form data submitted.
        data = request.form
        # retrieve the value corresponding to the key 'question' from the data dictionary
        question = data['question']
        # retrieve the value corresponding to the key 'reference' from the data dictionary
        reference = data['reference']
        model_name = data.get('model_name', DEFAULT_MODEL_NAME)

        # if a new model name is provided it loads the corresponding model 
        # To avoid loading the model and tokenizer every time, we only do it if the model name has changed
        # This condition checks whether the requested model name (model_name) is different from the last used model name (app.last_used_model_name)
        if model_name != app.last_used_model_name:
            # Load the new model and tokenizer
            app.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            app.tokenizer = AutoTokenizer.from_pretrained(model_name)
            app.last_used_model_name = model_name

        # Get the answer to the question
        # passes the retrieved data to answer_question function to get answer 
        answer = answer_question(question, reference, app.model, app.tokenizer)
        # if answer obtained it is returned as a JSON object otherwise default message returned 
        answer = answer if answer else 'I do not know the answer to that question 😢'

        # Return the predicted answer as a JSON object
        return jsonify({'answer': answer.capitalize()})
    else:
        # for get requests it renders the HTML template which contains form for inputting questions & reference text 
        # Return the HTML page with the form
        return render_template('index.html')


if __name__ == '__main__':
    app.run()
