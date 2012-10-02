<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Gnome</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="">
    <meta name="author" content="">

    <style>
        body {
            padding-top: 60px;
        }

        div#content {
            margin-bottom: 10px;
        }
    </style>

    <link href="/static/css/bootstrap.css" rel="stylesheet">
    <link href="/static/css/bootstrap-responsive.css" rel="stylesheet">
    <link href="/static/css/smoothness/jquery-ui-1.8.24.custom.css" rel="stylesheet">
    <%block name="css"> </%block>
    <link rel="shortcut icon" href="ico/favicon.ico">
</head>

<body>

<div class="navbar navbar-fixed-top">
    <div class="navbar-inner">
        <div class="container">
            <a class="btn btn-navbar" data-toggle="collapse" data-target=".nav-collapse">
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
            </a>
            <a class="brand" href="#">Gnome</a>
            <div class="nav-collapse collapse">
                <ul class="nav">
                    <li class="active"><a href="/">Home</a></li>
                    <li><a href="/model">Model</a></li>
                </ul>
            </div><!--/.nav-collapse -->
        </div>
    </div>
</div>

<%block name="content">
</%block>

<footer>
    <div class="container">
        <p>Copyright NOAA 2012.</p>
    </div>
</footer>

##<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.8.0/jquery.min.js"></script>
<script src="/static/js/jquery.1.8.0.min.js"></script>
<script src="/static/js/jquery-ui-1.8.24.custom.min.js"></script>
<script src="/static/js/bootstrap.min.js"></script>
<script src="/static/js/namespace.js"></script>
<%block name="javascript"> </%block>
</body>
</html>
