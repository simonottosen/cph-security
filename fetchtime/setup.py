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
                "ams-fetch=app.app:amsterdam",
                "lhr-fetch=app.app:heathrow",
                "muc-fetch=app.app:munich",
                "ist-fetch=app.app:istanbul",
                "edi-fetch=app.app:edinburgh",
                "dub-fetch=app.app:dublin"
            ]
        }
)
