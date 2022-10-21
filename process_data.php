<?php 

$filename = $_FILES["raw"]["name"];
$tempname = $_FILES["raw"]["tmp_name"]; 
$name1 =  time().'raw.png';
$folder1 = 'upload/'. $name1;   
if (move_uploaded_file($tempname, $folder1)) {
    $msg = "Image uploaded successfully";
}

$filename = $_FILES["mask"]["name"];
$tempname = $_FILES["mask"]["tmp_name"];
$name2 =   time().'mask.png';
$folder2 = 'upload/'.$name2;   
if (move_uploaded_file($tempname, $folder2)) {
    $msg = "Image uploaded successfully";
}

$command = "python api.py $name2 $name1";
$result = shell_exec($command);
$data = json_decode($result,true);
echo json_encode(array(
    'status' => 200,
    'data'  => $data,
    'command' => $command 
));
?>