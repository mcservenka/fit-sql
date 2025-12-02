# Multi-Type Benchmarking and FIT-SQL Evaluation

State-of-the-Art Text-to-SQL benchmarks represent one-dimensional challenges for modern LLMs, assuming answerability for every question in a dataset. However, in practical applications Text-to-SQL systems are required to handle a wide range of user inputs. This repository provides the source code for the paper **Text-to-SQL Under Realistic Conditions: Multi-Type Benchmarking and FIT-SQL Evaluation**, which addresses this gap by introducing a deterministic, schema-grounded framework for generating multiple question types and proposing FIT-SQL, a unified metric combining type classification and response quality. Augmented versions of Spider and BIRD-SQL are evaluated using leading LLMs. In order to reproduce the results, follow the instructions below.

>Cservenka, Markus. "Text-to-SQL Under Realistic Conditions: Multi-Type Benchmarking and FIT-SQL Evaluation", 2025.

Link to paper following soon...

## Ressources
To set up the environment, start by downloading the development sets of [Spider](https://yale-lily.github.io/spider) and [BIRD-SQL](https://bird-bench.github.io/) to the folders `./data/datasets/spider/` and `./data/datasets/bird/dev/` respectively. Then add the git submodule [Test-Suite-Evaluation](https://github.com/taoyds/test-suite-sql-eval) to  `./external/` and copy the BIRD's evaluation module `[evaluation.py](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py)` into `./external/bird/`. We will need them later for computing the execution accuracy of answerable samples. Make sure to define the OpenAI, Google and TogetherAI API keys in your environment variables as `OPENAI_API_KEY`, `OPENAI_API_ORGANIZATION`, `OPENAI_API_PROJECT` and `GOOGLE_API_KEY`, `TOGETHERAI_API_KEY`. We also recommend using the `dotenv`-package.

## Environment Setup
Now set up the Python environment:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Experiment
Follow the steps down below to recreate the experiment.

### Schema Representation
First you need to build the schema representation objects using `prepare_schema.py`. This will store each database schema of both datasets, Spider and BIRD, as a json-file in `data/schemas/`
```
python prepare_schema.py --dataset spider
```

### Generate Questions
Next we can start augmenting the original datasets by incorporating unanswerable, schema-based ambiguous and improper user inputs. You can run `generate_questions.py` to generate these questions for a specific dataset `{"spider", "bird"}`:
```
python generate_questions.py --dataset spider
```
### Prompt Model
Now using our augmented versions of Spider and BIRD, we can start prompting our models. In terms of LLMs utilized within this study, open- and closed-source LLMs were tested, which is common in this field of research. The first category consists of the models `qwen-3-80B` and `llama-3.3-70B` provided by TogetherAI (`together`). For close-source models we selected `gpt-5` and `gemini-2.5-pro` by OpenAI (`openai`) and Google (`google`) respectively. To generate the individual results in `data/reesults/` run the following command for each model and dataset:
```
python prompt_model.py --dataset spider --model gpt-5
```

### Evaluate Results
Eventually, you can evaluate the responses by running `evaluate_results.py`. This will add various evaluation scores (FIT-SQL, Classification Score, Response Score) to your response objects and create a new file in the form of `data/results/<dataset>/<model>_eval.json`. Please refer to the original paper for the definition of each metric.
```
python evaluate_results.py --dataset spider --model gpt-5
```

## Experiment Results
Down below we illustrated the official results of our paper. Please note that the results may vary after rerunning the experiment due to the inherent stochasticity of the LLM. For detailed evaluation results feel free to check out chapter 7 of the paper.

<table>
    <tr>
        <td> FIT-SQL by Model </td>
        <td> Spider </td>
        <td> BIRD </td>
    </tr>
    <tr>
        <td> GPT-5 </td>
        <td> 87.0 % </td>
        <td> 59.7 % </td>
    </tr>
    <tr>
        <td> Gemini-2.5-Pro </td>
        <td> 89.8 % </td>
        <td> 66.8 % </td>
    </tr>
    <tr>
        <td> Qwen-3-80B </td>
        <td> 72.1 % </td>
        <td> 43.9 % </td>
    </tr>
    <tr>
        <td> Llama-3.3-70B </td>
        <td> 82.4 % </td>
        <td> 57.1 % </td>
    </tr>
</table>

## Citation
```citation
@article{fit-sql,
    author  =   {Cservenka Markus},
    title   =   {Text-to-SQL Under Realistic Conditions: Multi-Type Benchmarking and FIT-SQL Evaluation},
    year    =   {2025}
}

