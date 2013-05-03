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
    <link href="/static/css/custom-theme/jquery-ui-1.8.16.custom.css" rel="stylesheet">
    <link href="/static/css/base.css" rel="stylesheet">

    <%block name="extra_head"> </%block>
    <link rel="shortcut icon" href="ico/favicon.ico">
</head>

<body>

##<div class="navbar navbar-fixed-top">
<div class="navbar">
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


<div class="container-fluid outer-wrapper expand">
    <div class="row-fluid inner-wrapper expand">
        <div class="hidden" id="sidebar">
            <div id="sidebar-container">
                <%block name="sidebar"> </%block>
            </div>
        </div>
        <div class="span8 expand" id="content">
            <%block name="content"> </%block>
        </div>
    </div>
</div>

<footer>
    <div class="container-fluid">
    </div>
</footer>

<%block name="javascript"> </%block>
</body>
</html>
