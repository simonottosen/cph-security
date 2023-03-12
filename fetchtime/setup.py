from setuptools import setup

setup(
    name="app",
    packages=["app"],
    include_package_data=True,
    zip_safe=False,
    entry_points={
            "console_scripts": [
                "fra-fetch=app.app:frankfurt",
                "dus-fetch=app.app:dusseldorf",
                "cph-fetch=app.app:copenhagen",
                "arn-fetch=app.app:arlanda",
                "osl-fetch=app.app:oslo",
                "ber-fetch=app.app:berlin"
            ]
        }
)