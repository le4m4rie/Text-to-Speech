# Course Website: Seminar Aktuelle Themen der künstlichen Intelligenz, 23S

## Meta-Information

*   Module Maintainer:  Florian Wahl (`@fwahl711`)
*   Course: [Seminar Aktuelle Themen der künstlichen Intelligenz](https://ki-seminar.github.io)
*   Institute: [Department of Applied Computer Science, Deggendorf University of Technology](https://th-deg.de/ai)
*   Current Version: [Summer 2023](https://ki-seminar.github.io/23s)


## Building the Site

We use [MkDocs]() to build the the course websites for this course.

* Install MkDocs with the following command:
```{.bash, id:"j29ie3c7"}
pip install mkdocs
```
* Install the style we use:
```{.bash, id:"j29ie3c7"}
pip install mkdocs-material
```
* Look at a copy of the site served locally on your machine:
```{.bash, id:"j29ie3c7"}
mkdocs serve
```
* We use a GitHub action to automatically deploy the website after pushing to the online repository

The idea for this course website is based on [https://github.com/pp4rs/2022-uzh/](https://github.com/pp4rs/2022-uzh/).
