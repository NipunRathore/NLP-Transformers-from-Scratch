# this py file contains of unit tests written using pytest framework 
import pytest
from app import app, DEFAULT_MODEL_NAME

# checks whether the default model, tokenizer, and last used model name are initialized correctly when the Flask application starts.
def test_default_model():
    # assures that the last used model name matches the default model name, and both the model and tokenizer objects are not None.
    assert app.last_used_model_name == DEFAULT_MODEL_NAME
    assert app.tokenizer is not None
    assert app.model is not None

# this test checks whether HTML page rendered correctly when GET request made to root URL '/'
def test_html_page():
    response = app.test_client().get('/')
    # asserts that the HTTP response status code is 200 (OK) & that the response data contains the expected text 'QnA'.
    assert response.status_code == 200
    assert 'QnA' in response.data.decode('utf-8')


# this test checks whether the form submission functionality works as expected 
def test_form_submission():
    # Submit the form with a question and reference text
    # simulates a POST request with question and reference and verifies response code is 200
    response = app.test_client().post('/', data={
        'question': 'What is the capital of France?',
        'reference': 'Paris is the capital of France.',
    })
    # then checks whether the expected JSON contains the answer 
    assert response.status_code == 200
    data = response.get_json()
    assert data['answer'] == 'Paris'

# checks the behavior when the form is submitted with empty input fields (question or reference text).
def test_empty_input():
    # Submit the form with an empty question
    # simulates POST request with empty question and response and verifies the response JSON contains expected default message 
    response = app.test_client().post('/', data={
        'question': '',
        'reference': 'France is a country in Europe.',
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['answer'] == 'I do not know the answer to that question 😢'

    # Submit the form with an empty reference text
    response = app.test_client().post('/', data={
        'question': 'What is the capital of France?',
        'reference': '',
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['answer'] == 'I do not know the answer to that question 😢'

# checks whether changing the model name in the form submission results in the expected behavior.
def test_model_name_change():
    # simulates a POST request with a different model name and verifies that the last used model name is updated accordingly, and both the model and tokenizer objects are not None.
    # Submit the form with a different model name
    response = app.test_client().post('/', data={
        'question': 'What is the capital of France?',
        'reference': 'France is a country in Europe.',
        'model_name': 'bert-base-cased',
    })
    assert response.status_code == 200
    assert app.last_used_model_name == 'bert-base-cased'
    assert app.model is not None
    assert app.tokenizer is not None

# this block included to run the tests when the script executed directly 
if __name__ == '__main__':
    pytest.main()