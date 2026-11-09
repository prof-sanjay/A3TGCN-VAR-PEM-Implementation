1. Please be connected to the Internet while compiling for the first time.
   Latex will dowload the required packages and configure itself. This will take some time.
   If you are using Overleaf, upload this folder and edit the file as follows:
2. TITLE PAGE Formating:

(a) You need to enter your  title of thesis in the appropriate space as illustrated below : 
 
\title{Enter The Project Title}     (Its not Case-sensitive)

(b) Name of the project students: (In the ascending order of Register Number)
\firstauthor{ Enter your name here}

(c) Other fields:
Other fields like guide's name, Guide's designation, HOD name, department, Branch of study/specialisation, Month & Year of submission must be entered in the respective fields. All these data will be automatically formated and written in the respective places with appropriate font size. 
3. ABSTRACT:
The student has to either type or copy & paste the contents of the abstract just below the section \abstract.

4. TABLE OF CONTENTS, LIST OF TABLES, LIST OF FIGURES:

The student need not do anything as all information required for the above tables will be automatically generated as the student adds the chapter-wise content.


5. LIST OF ABBREVIATIONS:
The student has to input all abbreviations under the section \abbreviations as per the below format:

\acro {acronym} {expanded form}

e.g., \acro {GA} {Genetic Algorithm}
      \acro {WHO} { World Health Organisation}



6. LIST OF AND SYMBOLS AND NOMENCLATURE:
The student has to input all symbols and nomenclature in alphabetical order under the section \chapter* {list of symbols} as per the below format:

\textbf {$\symbol$}   \>what the symbol represents, name of the unit

e.g.,\textbf {$\theta$}   \>Angle of twist, rad
     \textbf {$\Sigma$}   \>Stress, N/sq.m


7. CHAPTERS:
The student has to type (if already has it in Word format the student may copy & paste) the contents of the chapter just below the section \chapter.
(a) To start a new section, use the syntax \section {section title}
     < Type your contents coming under this section>
(b) To start a sub-section and a sub-sub-section, you can use the syntax
  \subsection {subsection title}
  \ subsubsection {subsubsection title}


8. TO INSERT A FIGURE, TABLE, FLOWHCART OR GRAPH, PLEASE REFER BACK TO THE .TEX FILE.


9. BIBLIOGRAPHY (OR) REFERENCE

The student has to download any of the reference management system softwares such as JabRef. It is a freeware which can be downloaded and installed.
Then create a database of all the references like journal articles,
confernce proceedings, Technical reports, Book, Book chapter, Web page etc in JabRef management system and have this file in the folder where your .tex file is placed.

At the end of the .tex template, there is a field \bibliography{Enter your reference management file (.bib)}. For cross reference the same in your text, call them by 
their key word, in this way \cite{key word}. 

Then the references which you cross referenced alone are formatted in your report as per University specifications.

Feel free to contact for any suggestions/ comments: ak_sumesh@cb.amrita,edu