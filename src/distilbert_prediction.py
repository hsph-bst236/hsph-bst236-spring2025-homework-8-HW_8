from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import re
import csv

# Load model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

# Load wiki.txt for contexts
wiki_data = {}
with open('wiki.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split('. ', 1)
            if len(parts) >= 2:
                name = parts[0]
                context = parts[1] if len(parts) > 1 else line
                wiki_data[name.lower()] = context

# Function to predict answer
def predict_answer(question, context):
        # TODO [part a]: Use the model to predict the answer to the question
        # Hint: You need to input boht question and context to tokenizer
        # Check the API document in https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad
        # Return the answer as a string
        ### YOUR CODE HERE ###
        pass
        ### END YOUR CODE ###


# Function to find context for a person
def find_context(person):
    person_lower = person.lower()
    for name, context in wiki_data.items():
        if person_lower in name:
            return context
    return None

# Process birth_dev.tsv
correct = 0
total = 0
processed_questions = []

with open('birth_dev.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) == 2:
            question, expected_answer = row
            
            # Extract person name from the question
            person = question.replace("Where was ", "").replace(" born?", "")
            
            # Find context for this person
            context = find_context(person)
            if context:
                # Predict answer
                prediction = predict_answer(question, context)
                
                # Check if prediction matches expected answer
                is_correct = expected_answer.lower() in prediction.lower()
                
                # Save results for this question
                processed_questions.append({
                    'question': question,
                    'context': context,
                    'expected': expected_answer,
                    'prediction': prediction,
                    'correct': is_correct
                })
                
                if is_correct:
                    correct += 1
                total += 1

# Print results
accuracy = correct / total if total > 0 else 0

# Open a file to save the results
with open('distilbert_qa_model_prediction.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(f"Processed {total} questions\n")
    output_file.write(f"Correct: {correct}\n")
    output_file.write(f"Accuracy: {accuracy:.2%}\n")

    # Print some examples
    output_file.write("\nExamples:\n")
    for i, q in enumerate(processed_questions[:5]):
        output_file.write(f"\nExample {i+1}:\n")
        output_file.write(f"Question: {q['question']}\n")
        output_file.write(f"Context: {q['context']}\n")
        output_file.write(f"Expected: {q['expected']}\n")
        output_file.write(f"Prediction: {q['prediction']}\n")
        output_file.write(f"Correct: {q['correct']}\n")

    # Print all wrong answers
    output_file.write("\nWrong Answers:\n")
    wrong_answers = [q for q in processed_questions if not q['correct']]
    for i, q in enumerate(wrong_answers):
        output_file.write(f"\nWrong Answer {i+1}:\n")
        output_file.write(f"Question: {q['question']}\n")
        output_file.write(f"Context: {q['context']}\n")
        output_file.write(f"Expected: {q['expected']}\n")
        output_file.write(f"Prediction: {q['prediction']}\n")
