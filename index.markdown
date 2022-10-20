---
layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>
<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <title>Learning and Retrieval from Prior Data for Skill-based Imitation Learning</title>


<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->
<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>


<!-- Global site tag (gtag.js) - Google Analytics -->

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Titillium Web","HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }

  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300;
  }

IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}
hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table
	{
	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


</head>

<body data-gr-c-s-loaded="true">



<div id="primarycontent">
<center><h1><strong>Learning and Retrieval from Prior Data for Skill-based Imitation Learning</strong></h1></center>
<center><h2>
<span style="font-size:25px;">
    <a href="http://snasiriany.me/" target="_blank">Soroush Nasiriany<sup>1</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="" target="_blank">Tian Gao<sup>1</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="https://ai.stanford.edu/~amandlek/" target="_blank">Ajay Mandlekar<sup>2</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="https://cs.utexas.edu/~yukez" target="_blank">Yuke Zhu<sup>1</sup></a>&nbsp;&nbsp;&nbsp;
    </span>
   </h2>
    <h2>
    <span style="font-size:25px;">
        <a href="https://www.cs.utexas.edu/" target="_blank"><sup>1</sup>The University of Texas at Austin</a>&nbsp;&nbsp;&nbsp;
        <a href="https://www.nvidia.com/en-us/research/" target="_blank"><sup>2</sup>NVIDIA Research</a>   
        </span>
    </h2>
    <h2>
    <span style="font-size:20px;">Conference on Robot Learning (CoRL), 2022</span>
    </h2>

<center><h2><span style="font-size:25px;"><a href="" target="_blank"><b>Paper</b></a> &emsp; <a href="" target="_blank"><b>Code Coming Soon!</b></a></span></h2></center>
<!-- <center><h2><a href="https://github.com/UT-Austin-RPL/maple" target="_blank">Code</a></h2></center> -->
<!-- <center><h2><a href="">Paper</a> | <a href="">Poster</a> | <a href="./src/bib.txt">Bibtex</a> </h2></center>  -->

<!-- <p> -->
<!--   </p><table border="0" cellspacing="10" cellpadding="0" align="center">  -->
<!--   <tbody> -->
<!--   <tr> -->
<!--   <\!-- For autoplay -\-> -->
<!-- <iframe width="560" height="315" -->
<!--   src="https://www.youtube.com/embed/GCfs3DJ4aO4?autoplay=1&mute=1&loop=1" -->
<!--   autoplay="true" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>   -->
<!--   <\!-- No autoplay -\-> -->
<!-- <\!-- <iframe width="560" height="315" -\-> -->
<!-- <\!--   src="https://www.youtube.com/embed/GCfs3DJ4aO4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>   -\-> -->

<!-- </tr> -->
<!-- </tbody> -->
<!-- </table> -->

<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
Imitation learning offers a promising path for robots to learn general-purpose tasks, but traditionally has enjoyed limited scalability due to high data supervision requirements and brittle generalization. Inspired by recent work on multi-task imitation learning, we investigate whether leveraging prior data from previous related tasks can enable learning novel tasks in a more robust, data-efficient manner. To make effective use of the prior data, the agent must internalize knowledge from the prior data and contextualize this knowledge in novel tasks. To that end we propose a skill-based imitation learning framework that extracts temporally-extended sensorimotor skills from prior data and subsequently learns a policy for the target task with respect to these learned skills. We find a number of modeling choices significantly improve performance on novel tasks, namely representation learning objectives to enable more predictable skill representations and a retrieval-based data augmentation procedure to increase the scope of supervision for the policy. On a number of multi-task manipulation domains, we demonstrate that our method significantly outperforms existing imitation learning and offline reinforcement learning approaches.
</p></td></tr></table>
</p>
  </div>
</p>

<hr>

<h1 align="center">Method Overview</h1>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <!-- <a href="./src/approach.png"> <img src="./src/approach.png" style="width:100%;">  </a> -->
  <video muted autoplay width="100%">
      <source src="./src/overview_animation.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>

</tbody>
</table>
  <table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  We present a skill-based imitation learning framework that uses prior data to learn novel tasks in a data-efficient manner. (left) First, we learn a latent skill model on the prior data. (middle) Given a dataset of target task demonstrations we use this latent space to retrieve similar behaviors from the prior data. (right) We then learn a policy on the target task data and retrieved prior data, trained to output latent skills.
</p></td></tr></table>


<br><hr> <h1 align="center" style="width:80%;">Skill-based Imitation Learning with Retrieval</h1>

<table width=800px><tr><td> <p align="justify" width="20%">
Our method consists of a skill learning and policy learning phase. (left) In the
skill learning phase we learn a latent skill representation of sub-trajectories via a variational autoencoder, and we include an additional temporal predictability term to learn a more consistent latent representation. (right) In the policy learning phase we train the policy to predict the latent skill given a history of observations preceding the sub-trajectory. To execute the policy we decode the predicted latent using the skill decoder.</p></td></tr></table>

<table border="0" cellspacing="10"
cellpadding="0" align="center"><tbody><tr><td align="center"
valign="middle"><img
src="./src/model.png" style="width:90%;"></td>
</tr> </tbody> </table>
<br>

<hr>

<!--
<h1 align="center">Simulated Environment Evaluation</h1>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>
  <p align="justify" width="20%">We perform evaluations on eight manipulation tasks. The first six come from the <a href="http://robosuite.ai/" target="_blank"> robosuite benchmark</a>, and we designed the last two (cleanup, peg insertion) to test our method in multi-stage, contact-rich tasks.</p>
</td></tr>
</tbody>
</table>
<video muted autoplay loop width="80%">
    <source src="./src/envs_animated_trimmed.mp4"  type="video/mp4">
</video>

<table width=800px><tr><td> <p align="justify" width="20%">
<br>
We compare MAPLE to five baselines, including a method that only employs low-level motor actions (<b>Atomic</b>), a state-of-the-art hierarchical DRL method that learns low-level options along with high-level controllers (<b>DAC</b>), methods that employ alternative policy modeling designs (<b>Flat</b> and <b>Open Loop</b>), and a self-baseline of our method that excludes low-level motor actions (<b>MAPLE (Non-Atomic)</b>).
We find that MAPLE significantly outperforms these baselines.
</p></td></tr></table>
<img src="./src/quant_results.png" style="width:80%;">

<table width=800px><tr><td> <p align="justify" width="20%">
<br> <br>
For reference, we visualize sample rollouts on the peg insertion task across all baselines:
</p></td></tr></table>
<video muted autoplay loop width="100%">
    <source src="./src/peg_insertion_cropped.mp4"  type="video/mp4">
</video>

<br>
<hr> <h1 align="center">Model Analysis</h1>

<table width=800px><tr><td> <p align="justify" width="20%">
We present an analysis of the task sketches that our method learned for each task.
Each row corresponds to a single sketch progressing temporally from left to right.
We see evidence that the agent unveils compositional task structures by applying temporally extended primitives whenever appropriate and relying on atomic actions otherwise.
For example, for the peg insertion task the agent leverages the grasping primitive to pick up the peg and the reaching primitive to align the peg with the hole in the block, but then it uses atomic actions for the contact-rich insertion phase.
<br>
<center><b>(click image to view full resolution)</b></center>
</p></td></tr></table>

<a href="./src/learned_sketches.png" target="_blank"> <img
src="./src/learned_sketches.png" style="width:100%;"> </a>

<br><hr>
<h1 align="center">Real-World Evaluation</h1>
<table width=800px><tr><td> <p align="justify" width="20%">
As our behavior primitives offer high-level action abstractions and encapsulate low-level complexities of motor actuation, our policies can directly transfer to the real world. We trained MAPLE on simulated versions of the stack and cleanup tasks and executed the resulting policies to the real world. Here we show rollouts on the cleanup task (played at 5x).
</p></td></tr></table>

<video muted controls width="80%">
    <source src="./src/cleanup_real.mp4"  type="video/mp4">
</video>

-->


<br>
<br>
<hr>
<center><h1>Citation</h1></center>

<table align=center width=800px>
              <tr>
                  <td>
                  <left>
<pre><code style="display:block; overflow-x: auto">
    @inproceedings{nasiriany2022learning,
      title={Learning and Retrieval from Prior Data for Skill-based Imitation Learning},
      author={Soroush Nasiriany and Tian Gao and Ajay Mandlekar and Yuke Zhu},
      booktitle={Conference on Robot Learning (CoRL)},
      year={2022}
    }
</code></pre>
</left></td></tr></table>
<br><br>

<div style="display:none">
<!-- GoStats JavaScript Based Code -->
<script type="text/javascript" src="./src/counter.js"></script>
<script type="text/javascript">_gos='c3.gostats.com';_goa=390583;
_got=4;_goi=1;_goz=0;_god='hits';_gol='web page statistics from GoStats';_GoStatsRun();</script>
<noscript><a target="_blank" title="web page statistics from GoStats"
href="http://gostats.com"><img alt="web page statistics from GoStats"
src="http://c3.gostats.com/bin/count/a_390583/t_4/i_1/z_0/show_hits/counter.png"
style="border-width:0" /></a></noscript>
</div>
<!-- End GoStats JavaScript Based Code -->
<!-- </center></div></body></div> -->
