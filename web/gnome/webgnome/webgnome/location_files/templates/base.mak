<%namespace name='defs' file='webgnome:templates/defs.mak'/>

<form class='wizard'>
    <div class="step" data-step=1>
        <h1><%block name='title'/></h1>

        <%block name='intro'/>

        <div class='references'>
            <%block name='references'/>
        </div>

        ${defs.references_btn()}
        ${defs.cancel_btn()}
        ${defs.next_btn()}
    </div>

    ${self.body()}
</form>

<%block name='javascript'/>
