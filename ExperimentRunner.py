import argparse
import json
import re
import subprocess
from comet_ml import Experiment

__author__ = "Manolomon"
__license__ = "MIT"
__version__ = "1.0"


def scrap_output(output, model_params):
    output = output.split('\n')
    step = 0
    for i in range(len(output)):
        if i % 2 == 0:  # Step and f(x)
            try:
                step = int(re.search('step (.+?):', output[i]).group(1))
                fx = float(re.compile(
                    r'(\d+\.\d+)$').search(output[i]).group(1))
                # print(str(step) + ": " + str(fx))
                experiment.log_metric(step=step, name='accuracy', value=fx)
            except AttributeError:
                break
        else:  # Params
            parse = output[i][7:-2]
            params = [float(i) for i in re.split(',', parse)]
            log_parameters = {}
            for entry in range(len(model_params)):
                log_parameters[model_params[entry]] = params[entry]
            # print(log_parameters)
            experiment.log_metrics(log_parameters, step=step)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description='Experiment logging and runner for the hyperparameter optimization process')
    ap.add_argument('--model', dest='model', type=str,
                    help='Algorithm top run (MultinomialNB, DecisionTreeClassifier, SVC, RandomForestClassifier)')

    args = ap.parse_args()

    with open('../configurations.json') as json_file:
        model_params = json.load(json_file)

    with open('main.py', 'r') as file:
        code = file.read()

    parameters = {k: model_params[args.model][k]
                  for k in ('population', 'generations', 'crossover', 'factor')}

    
    experiment = Experiment(
        api_key='ali1oQ27eFqaZYgKZmCtV3sfK',
        project_name='quality-attributes',
        log_code=False,
        auto_param_logging=False
    )
    experiment.add_tag(args.model)
    experiment.set_code(code=code, overwrite=True)

    experiment.log_parameters(parameters)

    proc = subprocess.Popen(
        "python main.py --np %s --max_gen %s --cr %s --f %s" %(
            parameters['population'],
            parameters['generations'],
            parameters['crossover'],
            parameters['factor'],
            ),
        stdout=subprocess.PIPE,
        shell=True)
    output = proc.stdout.read()

    scrap_output(output.decode("utf-8"), model_params[args.model]['hyperparameters'])