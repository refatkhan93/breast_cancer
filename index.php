<?php

// $result = shell_exec("python api.py mask.png raw.png");
// $data = json_decode($result,true);
// print_r($data);

?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Breast Cancer Detection</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"
        integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.5.2/css/bootstrap.css">
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="banner">
        <div class="header">
            <p class="textinside"><b>Breast Cancer</b> detection from ultrasound images</p>
        </div>
    </div>
    <div class="container card">
        <form action="" class="form" id="process_form" enctype="multipart/form-data">
            <div class="row">
                <div class="col-md-6">
                    <label for="">Image</label>
                    <input type="file" class="form-control" name="raw" >
                </div>
                <div class="col-md-6">
                    <label for="">Mask</label>
                    <input type="file" class="form-control" name="mask" >
                </div>
                <div class="col-md-12">
                    <button type="submit" style="width:100%; margin-top:10px;" class="btn btn-success"  id="submitbtn">Process</button>
                </div>
            </div>
        </form>
        <div class="row">
            <div class="col-md-6" id="result" style="margin-top: 10px;"></div>
        </div>
        <div class="row">
            <div class="col-md-6">
                <div id="raw_process" class="process"></div>
            </div>
            <div class="col-md-6">
                <div id="mask_process" class="process"></div>
            </div>
        </div>
    </div>
</body>
</html>
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js"
    integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous">
</script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"
    integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous">
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"
    integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous">
</script>
<script src="https://code.jquery.com/jquery-3.5.1.js"></script>
<script>
$(document).ready(function() {
    $("#process_form").on('submit',function(e) {
        e.preventDefault();
        $.ajax({
		url: "process_data.php",
		type: "POST",
		cache: false,
        data: new FormData(this),
        dataType: 'json',
        contentType: false,
        cache: false,
        processData:false,
		beforeSend: function () {
			$("#submitbtn").attr("disabled", "disabled");
			$("#submitbtn").html('Processing <i class="fa fa-spinner fa-spin"></i>');
		},
		success: function (data) {
            $("#submitbtn").removeAttr("disabled");
            $("#result").html('<p>Detected Result: <b>'+data.data["val"]+'</b></p>');
            $("#raw_process").html('<img class="imgcls" src="processed_image/'+data.data["raw"]+'" /> <p style="text-align:center;">Processed Raw Image</p>');
            $("#mask_process").html('<img class="imgcls" src="processed_image/'+data.data["mask"]+'" /> <p style="text-align:center;">Processed Mask</p>');
            // $("#mask_process").html('<img src="' + this.href + '" />');
		}
	    });
    });
});
</script>