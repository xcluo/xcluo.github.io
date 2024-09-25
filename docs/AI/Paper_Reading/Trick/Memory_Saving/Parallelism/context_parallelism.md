https://www.bilibili.com/video/BV1gb421771Z/?spm_id_from=333.337.search-card.all.click&vd_source=782e4c31fc5e63b7cb705fa371eeeb78

https://arxiv.org/pdf/2310.01889

attention计算复杂度与seq_len平方成正比，直接超长训练context window压力过大，可通过context parallelism提升计算性能


1. CP：计算流程图
<div class="one-image-container">
    <!-- <img src="\AI\Paper_Reading\Trick\Memory_Saving\Parallelism\image\without_cp.jpg" style="width: 100%;"> -->
    <!-- <p style="text-align: center;">图片标题</p> -->

    <img src="\AI\Paper_Reading\Trick\Memory_Saving\Parallelism\image\with_cp.jpg" style="width: 100%;">
    <p style="text-align: center;">CP communication pipeline. (AG: all gather; RS: reduce scatter)</p>
</div>
1. ring CP：key-value blocks traverse through a ring of hosts示意图
<div class="one-image-container">
    <img src="\AI\Paper_Reading\Trick\Memory_Saving\Parallelism\image\ring-based_cp.jpg" style="width: 80%;">
    <p style="text-align: center;">P2P只传递K,V（一般搭配GQA一起使用，此时$\#K,\#V \ll \#Q$）给临近的下一张GPU</p>
</div>
1. ring CP：ring attention calculation示意图
<div class="one-image-container">
    <img src="\AI\Paper_Reading\Trick\Memory_Saving\Parallelism\image\ring_cp_steps.gif" style="width: 50%;">
    <p style="text-align: center;">ring attention calculation</p>
</div>