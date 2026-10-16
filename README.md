# eyetrackprep workflow

## Description

`eyetrackprep` is a library to format and process pupil and gaze data acquired with eye-tracking alongside fMRI and physiological data for the Courtois project on Neural Modelling (CNeuroMod).

```text
Raw pupil and gaze files (restricted due to sensitive content)
      ↓
Dataset overview (QC report)
      ↓
BIDS formatting and metadata (Open-access BIDS repository)
      ↓
(Optional) Gaze drift correction (Open-access derivative repository)
      ↓
Plots to support offline quality assessments
```

Additional scripts that remove sensitive information (e.g., scan dates) and perform data visualisation (gaze density plots per subject, session, run or trial; gaze projection onto video stimuli) are also included. 

## Quick start

```
git clone git@github.com:courtois-neuromod/eyetrackprep.git

python3 -m venv <venv_name>
source <venv_name>/bin/activate

cd eyetrackprep
pip install -r requirements.txt
```

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="25%"><a href="https://github.com/MarieStLaurent"><img src="https://avatars.githubusercontent.com/u/9339621?v=4?s=200" width="200px;" alt="<Marie Saint-Laurent"/><br /><sub><b>Marie Saint-Laurent</b></sub></a><br /><a href="#ideas-MarieStLaurent" title="Ideas, Planning, & Feedback">🤔</a> <a href="#code-MarieStLaurent" title="Code">💻</a> <a href="#review-MarieStLaurent" title="Review">👀</a> <a href="#maintenance-me-pic" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="25%"><a href="https://github.com/me-pic"><img src="https://avatars.githubusercontent.com/u/77584086?v=4?s=200" width="200px;" alt="Marie-Eve Picard"/><br /><sub><b>Marie-Eve Picard</b></sub></a><br /><a href="#ideas-me-pic" title="Ideas, Planning, & Feedback">🤔</a> <a href="#review-me-pic" title="Review">👀</a> </td>
    </tr>
  </tbody>
</table>

## Acknowlegments
