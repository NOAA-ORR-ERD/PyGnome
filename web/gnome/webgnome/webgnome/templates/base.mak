<%namespace name="defs" file="defs.mak"/>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Gnome</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="">
    <meta name="author" content="">

    <link href="/static/css/bootstrap.css" rel="stylesheet">
    <link href="/static/css/bootstrap-responsive.css" rel="stylesheet">
    <link href="/static/css/smoothness/jquery-ui-1.8.24.custom.css" rel="stylesheet">
    <link href="/static/css/base.css" rel="stylesheet">
    <%block name="css"> </%block>
    <link rel="shortcut icon" href="ico/favicon.ico">
</head>

<body>

<div class="navbar navbar-fixed-top">
    <div class="navbar-inner">
        <div class="container-fluid">
           <a class="btn btn-navbar" data-toggle="collapse" data-target=".nav-collapse">
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
            </a>
            <a class="brand" href="/">WebGNOME</a>
            <div class="nav-collapse collapse">
                <%block name="navbar"></%block>
            </div><!--/.nav-collapse -->
        </div>
    </div>
</div>


<div class="container-fluid" id="content">
    <div class="row-fluid">
        <div class="span2" id="sidebar">
            <%block name="sidebar"> </%block>
        </div>
        <div class="span8">
            <%block name="content"> </%block>
        </div>
    </div>
</div>

<footer>
    <div class="container-fluid">
        <p>Copyright NOAA 2012.</p>
    </div>
</footer>

## Enable this to use Google as a CDN for jQuery instead of serving it ourselves.
##<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.8.0/jquery.min.js"></script>
<script src="/static/js/jquery.1.8.0.min.js"></script>
<script src="/static/js/jquery-ui-1.8.24.custom.min.js"></script>
<script src="/static/js/bootstrap.min.js"></script>
<script src="/static/js/namespace.js"></script>
<script src="/static/js/util.js"></script>
<%block name="javascript"> </%block>
</body>
</html>
