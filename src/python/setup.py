from pathlib import Path
from zensols.pybuild import SetupUtil

su = SetupUtil(
    setup_path=Path(__file__).parent.absolute(),
    name="zensols.sdoh",
    package_names=['zensols', 'resources'],
    # package_data={'': ['*.html', '*.js', '*.css', '*.map', '*.svg']},
    package_data={'': ['*.conf', '*.json', '*.yml']},
    description='A model that predicts Social Determinants of Health.',
    user='plandes',
    project='sdoh',
    keywords=['tooling'],
    # has_entry_points=False,
).setup()
