Modified from <a href="https://github.com/microsoft/UDA">microsoft/UDA </a>. <br>


<h2>Introduction</h2>
Some transfer learning and partial transfer learning algorithms were implemented using PyTorch. There are 5 algorithms including ERM, CORAL, DAN (MMD), DANN, and IWAN (partial transfer learning) in total. Note that the author mainly researched fault diagnosis, so the default dataset is the DDS dataset constructed by ourselves. See more information in the next section.

<h2>Usage</h2>
You can directly modify the params in the function <font color='green'> parse_args</font> in <font color='green'> main.py</font>, then run main.py.

<h2>Using your own dataset</h2>
You should split your data into a train/test set as follows:<br>
&emsp;&emsp;├─x_train.pt<br>
&emsp;&emsp;├─y_train.pt<br>
&emsp;&emsp;├─x_test.pt<br>
&emsp;&emsp;└─y_test.pt<br>
Then write the data loader file refering to the file in <font color="green"> dds.py</font> under <font color='green'> datasets</font> folder. <br>
The file <font color='green'> task.py</font> under the folder <font color='green'> tasks</font> are set to generate transfer tasks, and our DDS dataset consists of 9 kinds of operation condition:
<table>
    <tr>
        <td>condition</td>
        <td>20R-0HP</td>
        <td>20R-4HP</td>
        <td>20R-8HP</td>
        <td>30R-0HP</td>
        <td>30R-4HP</td>
        <td>30R-8HP</td>
        <td>40R-0HP</td>
        <td>40R-4HP</td>
        <td>40R-8HP</td>
    </tr>
    <tr>
        <td>number</td>
        <td>0</td>
        <td>1</td>
        <td>2</td>
        <td>3</td>
        <td>4</td>
        <td>5</td>
        <td>6</td>
        <td>7</td>
        <td>8</td>
    </tr>
</table>
So there is a total of 72 transfer tasks. You can refer to the aforementioned file to generate your tasks. The shape of the 1-dimensional data is 1024 and the shape of the 2-dimensional data is 64x64. If you use the 2-dimensional data and use resnet18 as the backbone, please modify the <font color='green'> _cnn_fidm</font> parameter to 512 in line 16 in file <font color='green'> base_model.py</font> under the folder models.
