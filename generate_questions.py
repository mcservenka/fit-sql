import argparse

from models.question_generator import QuestionGenerator

"""

    generates questions and on the basis of schemas
    and stores them as json files in data/questions

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["spider", "bird"], default="spider")
    args = parser.parse_args()

    print(f"Starting question generation for {args.dataset}.")

    qg = QuestionGenerator(dataset=args.dataset)

    for db in qg.distinct_dbs:
        qg.generate_improper(db)
        qg.generate_unanswerable(db)
        qg.generate_ambiguous(db)

    qg.calculate_statistics()
    qg.save_questions_json()