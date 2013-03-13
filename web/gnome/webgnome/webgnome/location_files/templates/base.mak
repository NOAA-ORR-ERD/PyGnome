<%namespace name='defs' file='webgnome:templates/defs.mak'/>


<%def name='height()'>350</%def>
<%def name='width()'>350</%def>
<%def name='references()'></%def>

<form class='wizard form page hide' id="${self.form_id()}" title="${self.title()}"
        data-height="${self.height()}" data-width="${self.width()}">
    <%defs:step>
        ${self.intro()}

        <div class='references' title="References">
            ${self.references()}
        </div>

        <%defs:buttons>
            ${defs.references_btn()}
            ${defs.next_btn()}
            ${defs.cancel_btn()}
        </%defs:buttons>
    </%defs:step>

    ${self.body()}

    <%defs:step>
        <p>You are ready to move to the Map Window to start your spill and run the
        model. Here are the tools you'll use:</p>

        <p>To set the location of your spill, you will use the Spill Tool or the
        overflight tools (the Spray Can and Eraser), pictured below.</p>

        <p>To run the model, you will use GNOME's run controls: Run, Pause and
        Step (pictured below).</p>

        <%defs:buttons>
            ${defs.back_btn()}
            ${defs.finish_btn()}
        </%defs:buttons>
    </%defs:step>
</form>

<%block name='javascript'/>
