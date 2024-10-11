import os

from Orcar.log_utils import get_logger
from Orcar.tracer import read_tracer_output

logger = get_logger(__name__)


def main():
    """
    test_tracer_output.json is from astropy__astropy-12907
    Expected output:
    INFO     {
        CodeInfo(keyword='separability_matrix', file_path='/astropy__astropy/astropy/modeling/separable.py'),
        CodeInfo(keyword='_compute_n_outputs', file_path='/astropy__astropy/astropy/modeling/separable.py'),
        CodeInfo(keyword='_separable', file_path='/astropy__astropy/astropy/modeling/separable.py'),
        CodeInfo(keyword='_cstack', file_path='/astropy__astropy/astropy/modeling/separable.py'),
        CodeInfo(keyword='_coord_matrix', file_path='/astropy__astropy/astropy/modeling/separable.py')
    }
    """

    sensitivity_list = ["separability_matrix", "CompoundModels"]
    json_relative_path = "./tests/test_tracer_output.json"
    json_absolute_path = os.path.abspath(json_relative_path)
    function_list = read_tracer_output(json_absolute_path, sensitivity_list)
    logger.info(function_list)


if __name__ == "__main__":
    main()
