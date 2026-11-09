%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This is a class for LaTeX2e that should format a report as per
% the BTech Thesis guidelines.  The page layout is set using the
% geometry package.  The double/singlespacing is setup using
% setspace.  Figure captions are set using the hang option of
% caption2.  natbib is used to do the citation using an
% author-year format. I've made an iitm.bst file that formats 
% the references as per the  guidelines.
%
% To use this class do the following:
%  \documentclass[options]{ugthesis} 
%
% Options:
%
%   PhD, MS, MTech or BTech -- generates the appropriate title page
%   and loads the report class.
%
%   synopsis -- Generates the title page for the synopsis.  This also
%   loads the article class instead of the report.
%  
% Example:
%
% \documentclass[PhD,synopsis]{ugthesis}
% \documentclass[MS]{ugthesis}
%
% IMPORTANT NOTICE:
% 
%   PLEASE DO NOT MESS WITH THE MARGINS AND GENERAL FORMATTING SETUP
%   IN THIS FILE UNLESS YOU ARE ABSOLUTELY SURE THAT THE FORMAT DOES NOT
%   MATCH WITH THE THESIS GUIDELINES.  FOR EXAMPLE, DO NOT CHANGE THE 
%   MARGINS AND SPACING JUST TO MAKE YOUR THESIS LARGER!
% 
% Notes:
%
%   * I am using as much of the Thesis guidelines for the spacing
%     and margins as I can.
%   * This class will assume a4paper with 12pt fonts.
%   * I am no TeXpert so feel free to clean up the mess.

% This is an Adapted version of thesis from SRM university.

% please email your comments: ak_sumesh@cb.amrita.edu
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{ugthesis}[2014/12/01 Class for Dissertations]

\newif\if@PhD
\newif\if@MS
\newif\if@MTech
\newif\if@BTech
\newif\if@Synopsis

\DeclareOption{PhD}{\@PhDtrue}
\DeclareOption{MS}{\@MStrue}
\DeclareOption{MTech}{\@MTechtrue}
\DeclareOption{BTech}{\@BTechtrue}
\DeclareOption{synopsis}{\@Synopsistrue}
\ProcessOptions\relax

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Load the correct class.
\if@Synopsis
    \LoadClass[12pt,a4paper]{article}
\else
    \LoadClass[12pt,a4paper]{report}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Geometry and page layout
\RequirePackage[%compat2,
a4paper,left=1.5in,right=1in,top=1in,bottom=1in,
                nohead]{geometry}[2002/07/08]
\parskip 8pt

\renewcommand{\@pnumwidth}{2.5em}
%\renewcommand{\@tocrmarg}{2.5em}
\newcommand{\bigsize}{\fontsize{16pt}{20pt}\selectfont}
\newcommand{\RomanNumeralCaps}[1]
    {\MakeUppercase{\romannumeral #1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Using setspace to do the single/double spacing.  Setspace also
%% makes all tables and figures have single spacing automatically.
\RequirePackage{setspace} 
\RequirePackage{atbeginend} % used to redefine the list environs
\setstretch{1.5}

\BeforeBegin{itemize}{\vspace*{-5ex}\begin{singlespace}}
\AfterEnd{itemize}{\end{singlespace}}

\BeforeBegin{enumerate}{\vspace*{-5ex}\begin{singlespace}}
\AfterEnd{enumerate}{\end{singlespace}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The figure captioning is done correctly via the hang option.
%%\RequirePackage[hang]{caption2}  -> TeX generates error
\RequirePackage[hang]{caption}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Bibliography
% 
% natbib allows easy switching of citation style and using it along
% with the iitm.bst makes things perfect.
\RequirePackage[authoryear,round]{natbib} 
%\RequirePackage[numbers,square]{natbib} 
\bibliographystyle{ascelike}

\newcommand{\bibstyle@ascelike}{\bibpunct{(}{)}{;}{a}{,}{,}%
     \gdef\NAT@biblabel##1{##1. }}
%
\setlength{\bibhang}{0pt}
\citestyle{ascelike}
%\citestyle{unsrt_newen}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Other useful macros.  Many of these are modifications from newdiss.cls

% \department{department name} -- user macro to declare the name of the department.
\def\guide#1{\gdef\@guide{#1}}
\def\hod#1{\gdef\@hod{#1}}
\def\designation#1{\gdef\@designation{#1}}
\def\department#1{\gdef\@department{#1}}
\def\guidedepartment#1{\gdef\@guidedepartment{#1}}
\def\firstauthor#1{\gdef\@firstauthor{#1}}
\def\firstauthorregno#1{\gdef\@firstauthorregno{#1}}
\def\secondauthor#1{\gdef\@secondauthor{#1}}
\def\secondauthorregno#1{\gdef\@secondauthorregno{#1}}
\def\thirdauthor#1{\gdef\@thirdauthor{#1}}
\def\thirdauthorregno#1{\gdef\@thirdauthorregno{#1}}
\def\fourthauthor#1{\gdef\@fourthauthor{#1}}
\def\fourthauthorregno#1{\gdef\@fourthauthorregno{#1}}
\def\fifthauthor#1{\gdef\@fifthauthor{#1}}
\def\fifthauthorregno#1{\gdef\@fifthauthorregno{#1}}
%\RequirePackage[dvips]{graphicx}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The title page

\def\maketitle{%
  \pagenumbering{roman}
  \thispagestyle{empty}
  \newpage
  \null
  \setcounter{page}{0}
  \parskip=0pt
  \begin{center}%
  \let \footnote \thanks
  \vglue 0in % this makes top margin 2in
  \vskip -5ex%
  \if@Synopsis
    {\large \em SYNOPSIS OF \par}
  \fi
    {\Large \bf \expandafter\MakeUppercase{\@title} \par}%
    \vskip 10ex%
    {\normalsize
      \if@PhD
         {\em A THESIS }\par
         \vskip 5ex%
         \if@Synopsis
            {\normalsize \em to be submitted by \par}
         \else
            {\normalsize \em submitted by \par}
         \fi
         \vskip 5ex%
         {\bf \large\MakeUppercase{\@firstauthor} {\@firstauthorregno}\par\par}
         \vskip 5ex%
         {\em for the award of the degree }\par
         \vskip 5ex%
         {\em of} \par
         \vskip 5ex%
         {\bf \large \MakeUppercase{Doctor of Philosophy}} \par
       \else
         \if@MS
            {\em A THESIS }\par
            \vskip 5ex%
            {\normalsize \em submitted by \par}
            \vskip 5ex%
            {\bf \MakeUppercase{\@firstauthor} {\@firstauthorregno} \par\par}
            \vskip 5ex%
            {\em for the award of the degree }\par
            \vskip 5ex%
            {\em of} \par
            \vskip 5ex%
            {\bf \large \MakeUppercase{Master Of Science}} \par
            (by Research) \par
         \else
{\large \bf \expandafter  24CS798 / 24AI798 -- DISSERTATION PHASE -- \RomanNumeralCaps{1}}\par
           \vskip 5ex%
           {\large \bf \em Submitted by \par}
           \if@BTech
           \begin{singlespacing}
            {\bf \bigsize\MakeUppercase{\@firstauthor} {\@firstauthorregno}\par}
            {\bf \bigsize\MakeUppercase{\@secondauthor} {\@secondauthorregno}\par}
            {\bf \bigsize\MakeUppercase{\@thirdauthor} {\@thirdauthorregno}\par}
            {\bf \bigsize\MakeUppercase{\@fourthauthor} {\@fourthauthorregno}\par}
            {\bf \bigsize\MakeUppercase{\@fifthauthor} {\@fifthauthorregno}\par}
            \end{singlespacing}
            \else
            {\bf \bigsize\MakeUppercase{\@firstauthor} {\@firstauthorregno}\par\par}
            \fi
                       \vskip 5ex%
        %\begin{singlespacing}
        %{\normalsize \em Under the guidance of \par}        
         %     {\bf \bigsize{\@guide}\par\par}
         %      {\normalsize{(\@designation, Department of \@guidedepartment)}\par\par}
        %\end{singlespacing}
           {\bf \large \em in partial fulfillment of the requirements for the degree
           \par
           of \par} \vskip 5ex%
                      \if@MTech
             {\bf \large MASTER OF TECHNOLOGY}
           \fi
           \if@BTech
              {\bf \large BACHELOR OF TECHNOLOGY}
           \fi
         \fi
      \fi
     {\par \bf \large \em in \par}
     {\bf \large \MakeUppercase{\@department}\par\par}
     %{\par of \par}
     
  \end{center}%

  %\vspace*{0.25in}
  %\centerline{\psfig{figure=emblem.ps}}
%  \centerline{\includegraphics{iitm.eps}}
  \centerline{\includegraphics[scale=1.25]{cert_logo.png}}
  %\vspace*{-0.15in}
  \begin{center}
  {\bf \large \MakeUppercase{DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING} \par\par}}
   \bf \large \MakeUppercase{AMRITA SCHOOL OF COMPUTING \\
 %\MakeUppercase{Amrita Vishwa Vidyapeetham}
  {\bf \large \MakeUppercase{Amrita Vishwa Vidyapeetham} \par\par}}
    {\large COIMBATORE -- 641 112 \par}
        \vglue 0.50em
    {\bf \large \@date}\par
  \end{center}
  %\parskip 8pt
}

\def\titlepage{\newpage\thispagestyle{empty}\c@page\z@}
\def\endtitlepage{\newpage}

\if@Synopsis
   \renewcommand\refname{REFERENCES}
\else
   % QUOTATIONS
   %
   \def\quotations{\chapter*{\centerline{QUOTATIONS}}
   \addcontentsline{toc}{chapter}{QUOTATIONS}
   }

   % DEDICATION
   %
   \def\dedication{\chapter*{\centerline{DEDICATION}}
   %\addcontentsline{toc}{chapter}{DEDICATION}
   }

   % CERTIFICATE
   %
   %   - this produces a certificate page as per IITM Regulations
   %     comment the following lines out to get it working.
   %
   \def\certificate{%% A preliminary chapter
     \clearpage
 
     \centerline{\LARGE \bf AMRITA VISHWA VIDYAPEETHAM}
  \centerline{\large \bf AMRITA SCHOOL OF COMPUTING, COIMBATORE}
  \centerline{\large \bf DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING}
  \vskip 25pt
\centerline{\includegraphics[scale=1]{cert_logo.png}}
    % \centerline{(Under Section 3 of UGC Act, 1956)}
      \vskip 25pt
     \centerline{\bigsize \bf BONAFIDE CERTIFICATE}
     \vskip 15pt
     \thispagestyle{empty}
     %\typeout{Certificate}
       \if@MTech
   \begin{doublespacing}

     \large This is to certify that the thesis entitled \ {\bf\MakeUppercase{\@title}}, submitted by  {\bf\MakeUppercase{\@firstauthor} {\@firstauthorregno}}, \normalfont in partial fulfillment of the requirements for the award of degree of \textbf{Master of Technology in Artificial Intelligence / Computer Science and Engineering}, is a bonafide record of the work carried out under my guidance and supervision at the Department of Computer Science and Engineering, Amrita School of Computing, Coimbatore.
    \end{doublespacing}
     \else
     \begin{doublespacing}
          {\large Certified that this project report titled ``{\bf\MakeUppercase{\@title}}'' is the bonafide work of ``{\bf\MakeUppercase{\@firstauthor} {\@firstauthorregno}}'', who carried out the project work under my supervision. Certified further, that to the best of my knowledge the work reported herein does not form any other project report or dissertation on the basis of which a degree or award was conferred on an earlier occasion on this or any other candidate.}
          \end{doublespacing}
          \fi
     \vspace*{1in}
     
     \begin{singlespacing}
     \hspace*{-0.25in}
     \parbox{2.5in}{
     \vskip 5ex%
     \noindent {\@guide} \\
     \noindent {Thesis Advisor} \\
     \noindent \@designation \\ 
     \noindent Department of {\@guidedepartment}\\
     \vskip 3ex%
     \noindent Evaluated on: \\
     \vskip 8ex%
    \hdashrule{4cm}{1pt}{1pt}\\
     \noindent Internal Examiner
     } 
     \hspace*{.9 in} 
     \parbox{2.5in}{
     \vskip 3ex%
     \noindent {\@hod} \\
     \noindent { Chairperson} \\
     \noindent Department of {\@guidedepartment}\\
     \vskip 21ex
     \hdashrule{4cm}{1pt}{1pt}\\
     \noindent External Examiner


     
     }  
     \end{singlespacing}
     \setcounter{page}{2}
   }
   % ACKNOWLEDGEMENTS
   %
   \def\acknowledgements{\chapter*{\centerline{ACKNOWLEDGEMENT}}
     \addcontentsline{toc}{chapter}{ACKNOWLEDGEMENT}
   }

% DECLARATION
   %
 \def\declaration{\chapter*{\centerline{DECLARATION}}
    % \addcontentsline{toc}{chapter}{DECLARATION}
We, the undersigned solemnly declare that the project report {\bf\MakeUppercase{\@title}} is based on our own work carried out during the course of our study under the supervision of \@guide, \@designation, \@guidedepartment, and has not formed the basis for the award of any other degree or diploma, in this or any other Institution or University. In keeping with the ethical practice in reporting scientific information, due acknowledgement have been made wherever the findings of others have been cited.
  \vskip 25pt
\noindent
          \@firstauthor {\@firstauthorregno}\\
           \@secondauthor{ \@secondauthorregno}\\
            \@thirdauthor{\@thirdauthorregno}\\
            \@fourthauthor {\@fourthauthorregno}\\
           \@fifthauthor {\@fifthauthorregno}
            
 }

   % ABSTRACT
   %   - this produces pages suitable for an IITM Abstract
   \def\abstract{\chapter*{\centerline{ABSTRACT}}
   \addcontentsline{toc}{chapter}{ABSTRACT}
    \thispagestyle{empty} }

   % GLOSSARY
   %
   \def\glossary{\chapter*{\centerline{GLOSSARY}}
   \addcontentsline{toc}{chapter}{GLOSSARY}
   }

   % ABBREVIATIONS
   %
   \def\abbreviations{\chapter*{\centerline{ABBREVIATIONS}}
   \addcontentsline{toc}{chapter}{Abbreviations}
   }

   % Use the nomencl package for the NOTATION section.

   % Makes the chapter heading
   \def\@makechapterhead#1{%
     {\parindent \z@ \raggedright \normalfont
       \ifnum \c@secnumdepth >\m@ne
       \begin{center}
         \Large\bfseries  \@chapapp\space \thechapter
       \end{center}
       \par\nobreak
       \vskip 20\p@
       \fi
       \interlinepenalty\@M
       \bigsize\bf
       \begin{center}
         \MakeUppercase{#1}\par\nobreak      
       \end{center}
       \vskip 20\p@
       %\thispagestyle{empty}
     }}

   % Makes the chapter heading for \chapter* commands
   \def\@makeschapterhead#1{%
     \vspace*{25\p@}%
     {\parindent \z@ \raggedright
       \normalfont
       \interlinepenalty\@M
       \Large \bfseries #1\par\nobreak
       \vskip 20\p@
     }}

   %LIST OF PAPERS
   \if@PhD
   \def\listofpapers{
           \normalbaselines
           \chapter*{\centerline{LIST OF PAPERS BASED ON THESIS}}
           \markboth{LIST OF PAPERS BASED ON THESIS}{LIST OF PAPERS BASED ON THESIS}
           %\addcontentsline{toc}{chapter}{LIST OF PAPERS BASED ON THESIS}
   }
      \fi
   % CV
   \def\CV{\chapter*{\centerline{CURRICULUM VITAE}}}

   % Comittee
   \if@PhD
       \def\committee{\chapter*{\centerline{DOCTORAL COMMITTEE}}}
   \else
       \if@MS
          \def\committee{\chapter*{\centerline{GENERAL TEST COMITTEE}}}}
        \fi
   \fi


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Renewed commands to set the titles of various pages correctly.
   \renewcommand\contentsname{\centering TABLE OF CONTENTS}
   \renewcommand\listfigurename{\centering LIST OF FIGURES}
   \renewcommand\listtablename{\centering LIST OF TABLES}
   \renewcommand{\chaptername}{Chapter}
   \renewcommand\bibname{\centering REFERENCES}
   \renewcommand\appendixname{APPENDIX}
\fi


