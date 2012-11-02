<%inherit file="base.mak"/>

<%block name="css">
    <link rel='stylesheet' type='text/css' href='/static/css/skin/ui.dynatree.css'>
    <link rel='stylesheet' type='text/css' href='/static/css/model.css'>
</%block>

<%block name="navbar">
    <ul class="nav" role="navigation">
        <li class="dropdown">
            <a id="file-drop" href="javascript:" role="button" class="dropdown-toggle" data-toggle="dropdown">Model<b class="caret"></b></a>
            <ul class="dropdown-menu" role="menu" aria-labelledby="file-drop">
                <li><a tabindex="-1" id="menu-new" href="javascript:">New</a></li>
                <li><a tabindex="-1" href="javascript:">Load from file</a></li>
                <li><a tabindex="-1" href="javascript:">Save</a></li>
                <li class="divider"></li>
                <li><a tabindex="-1" href="javascript:">Preferences</a></li>
            </ul>
        </li>
        <li class="dropdown">
            <a id="run-drop" href="javascript:" role="button" class="dropdown-toggle" data-toggle="dropdown">Run<b class="caret"></b></a>
            <ul class="dropdown-menu" role="menu" aria-labelledby="run-drop">
                <li><a tabindex="-1" id="menu-run" href="javascript:">Run</a></li>
                <li><a tabindex="-1" id="menu-step" href="javascript:">Step</a></li>
                <li><a tabindex="-1" id="menu-run-until" href="javascript:">Run Until...</a></li>
            </ul>
        </li>
        <li class="dropdown">
            <a id="help-drop" href="javascript:" role="button" class="dropdown-toggle" data-toggle="dropdown">Help<b class="caret"></b></a>
            <ul class="dropdown-menu" role="menu" aria-labelledby="help-drop">
                <li><a tabindex="-1" target="window" href="javascript:">About</a></li>
                <li><a tabindex="-1" href="javascript:">Tutorial</a></li>
            </ul>
        </li>
    </ul>
</%block>

<%block name="sidebar">
     <div class="container" id="sidebar-toolbar">
      <div class="btn-toolbar">
        <div class="btn-group">
            <a class="btn" id="add-button" href="javascript:"><i class="icon-plus-sign"></i></a>
            <a class="btn disabled" id="remove-button" href="javascript:"><i class="icon-minus-sign"></i></a>
            <a class="btn disabled" id="settings-button" href="javascript:"><i class="icon-wrench"></i></a>
        </div>
      </div>
    </div>
    <div id="tree"> </div>
</%block>

<%block name="content">
    <div class="container">
      <div class="messages">
          <div class="alert alert-success ${'' if success else 'hidden'}">
              <button type="button" class="close" data-dismiss="alert">× </button>
              <span class='message'>${success if success else ''}</span>
          </div>
          <div class="alert alert-warning ${'' if warning else 'hidden'}">
              <button type="button" class="close" data-dismiss="alert">× </button>
              <strong>Warning!</strong> <span class="message">${warning if warning else ''}</span>
          </div>
           <div class="alert alert-error ${'' if error else 'hidden'}">
              <button type="button" class="close" data-dismiss="alert">× </button>
              <strong>Error!</strong> <span class="message">${error if error else ''}</span>
          </div>
      </div>
    </div>

    <div class="btn-toolbar">
          <div class="btn-group">
              <a class="btn" id="fullscreen-button" href="javascript:"><i class="icon-fullscreen"></i></a>
          </div>
          <div class="btn-group">
              <a class="btn" id="resize-button" href="javascript:"><i class="icon-resize-small"></i></a>
          </div>
          <div class="btn-group">
            <a class="btn disabled" id="zoom-in-button" href="javascript:"><i class="icon-zoom-in"></i></a>
            <a class="btn disabled" id="zoom-out-button" href="javascript:"><i class="icon-zoom-out"></i></a>
            <a class="btn disabled" id="move-button" href="javascript:"><i class="icon-move"></i></a>
        </div>
        <div class="btn-group">
            <a class="btn disabled" id="back-button" href="javascript:"><i class="icon-fast-backward"></i></a>
            <div class="btn disabled" id="slider-container"><span id="time">00:00</span> <div id="slider"></div></div>
            <a class="btn" id="play-button" href="javascript:"><i class="icon-play"></i></a>
            <a class="btn disabled" id="pause-button" href="javascript:"><i class="icon-pause"></i></a>
            <a class="btn disabled" id="forward-button" href="javascript:"><i class="icon-fast-forward"></i></a>
        </div>
    </div>

    <div id="map">
    </div>

    <div id="placeholder" class="hidden">
        <img class="frame active" src="/static/img/placeholder.gif">
    </div>

    <div id="modal-container">
      <%include file="forms/run_model_until.mak" args="form=run_model_until_form, action_url=run_model_until_form_url"/>
      <%include file="forms/model_settings.mak" args="form=settings_form, action_url=settings_form_url"/>
      <%include file="forms/add_mover.mak" args="form=add_mover_form, action_url=add_mover_form_url"/>
      <%include file="forms/constant_wind_mover.mak" args="form=constant_wind_form, action_url=constant_wind_form_url"/>
      <%include file="forms/variable_wind_mover.mak" args="form=variable_wind_form, action_url=variable_wind_form_url"/>
    </div>
</%block>

<%block name="javascript">
    <script src='/static/js/jquery.imagesloaded.min.js' type="text/javascript"></script>
    <script src='/static/js/jquery.cycle.all.latest.js' type="text/javascript"></script>
    <script src='/static/js/jquery.cookie.js' type="text/javascript"></script>
    <script src="/static/js/jquery.dynatree.min.js"></script>
    <script src="/static/js/underscore-min.js"></script>
    <script src="/static/js/backbone-min.js"></script>
    <script src="/static/js/gnome.js"></script>

    <script type="text/javascript">
        $('#map').imagesLoaded(function() {
            new window.noaa.erd.gnome.AppView({
                mapEl: '#map',
                mapPlaceholderEl: '#placeholder',
                sidebarEl: '#sidebar',
                formContainerEl: '#modal-container',
                generatedTimeSteps: ${generated_time_steps_json or '[]' | n},
                expectedTimeSteps: ${expected_time_steps_json or '[]' | n},
                currentTimeStep: ${model.current_time_step},
                forms: ${form_urls | n}
            });
        });
    </script>
</%block>