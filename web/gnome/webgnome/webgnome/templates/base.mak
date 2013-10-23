<%namespace name="defs" file="defs.mak"/>

<!DOCTYPE html>
<html lang="en">
	<head>
	    <meta charset="utf-8">
	    <title>Gnome</title>
	    <meta name="viewport" content="width=device-width, initial-scale=1.0">
	    <meta name="description" content="">
	    <meta name="author" content="">

		<%block name="third_party_css"/>

		<%block name="third_party_js"/>

	    <link href="/static/css/base.css" rel="stylesheet">

	    <%block name="extra_head"/>

	    <link rel="shortcut icon" href="ico/favicon.ico">
	</head>

	<body>

		<nav class="navbar navbar-default" role="navigation">
	        <div class="navbar-header">
	           <button type="button" class="navbar-toggle" data-toggle="collapse" data-target=".navbar-ex1-collapse">
	           	    <span class="sr-only">Toggle navigation</span>
	                <span class="icon-bar"></span>
	                <span class="icon-bar"></span>
	                <span class="icon-bar"></span>
	            </button>

	            <a class="navbar-brand" href="#">WebGNOME</a>

	        </div>

            <%block name="navbar"/>
		</nav>

		<div class="container">
		    <div class="row">
		        <div class="hidden" id="sidebar">
		            <div id="sidebar-container">
		                <%block name="sidebar"/>
		            </div>
		        </div>
		        <div class="panel" id="content">
		            <%block name="content"/>
		        </div>
		    </div>
		</div>

		<footer>
		    <div class="container">
		    </div>
		</footer>

		<%block name="javascript"/>
	</body>
</html>
