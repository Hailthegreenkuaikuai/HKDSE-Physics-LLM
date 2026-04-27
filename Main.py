"""
Program Entry Point
"""
from Evaluate import *
from Utils import *
import datetime

MODEL = PRETRAINED_MODEL
EVAL = False
OLLAMA_ENABLE = False

if __name__ == "__main__":
    Path("Result").mkdir(parents=True, exist_ok=True)
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{MODEL}_{start_time}"
    with open(f"Result/{filename}_result.txt", "w") as f:
        f.write(f"Start Time: {start_time}\n")
    try:
        print(f"Loading {MODEL} and tokenizer")
        model, generator, tokenizer = load_model_and_tokenizer(MODEL)
        dataset = load_dataset(TEST_DATA)
        #dataset = dataset[:5] #For testing only
        if OLLAMA_ENABLE:
            print("Enable tools calling")
            generated_responses, references = ollama_responses_generation(dataset, MODEL)
        else:
            print("Disable tools calling")
            generated_responses, references = responses_generation(generator, tokenizer, dataset)
        responses = output_responses(generated_responses=generated_responses, references=references, dataset=dataset)
        with open(f"Generated_Responses/{filename}.txt", "w", encoding='utf-8') as g:
            g.write(responses)
        g.close()
        if EVAL:
            print("Enable evaluation")
            clean_responses = strip_thought(generated_responses)
            #print(generated_responses)
            #print(clean_responses)
            #PandaLMEvaluate()
            metrics, P, R, F1 = BERTScore(clean_responses, references)
            bleurt_score = BLEURT_Score(clean_responses, references)
            hallucination_score = SelfCheckGPT(generator, 20)
            perplexity_score = PerplexityScore(input_texts=clean_responses, tokenizer=tokenizer, model=model, device=DEVICE)
            result = output_result(BERTScore=metrics, BLEURT=bleurt_score, SelfCheckGPT=hallucination_score, Perplexity=perplexity_score)
            with open(f"Result/{filename}_result.txt", "a", encoding='utf-8') as f:
                start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                f.write(result)
            f.close()
    finally:
        end_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"Result/{filename}_result.txt", "a", encoding='utf-8') as f:
            f.write(f"\nEnd Time: {end_time}\n")
        f.close()