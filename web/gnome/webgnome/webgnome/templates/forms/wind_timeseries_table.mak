<div class="wind-values">
    <table class="table table-striped time-list">
        <thead>
        <tr class='table-header'>
            <th>Date (m/d/y)</th>
            <th>Time</th>
            <th>Speed</th>
            <th>Wind From</th>
            <th>&nbsp;</th>
        </tr>
        </thead>
        <tbody>
        </tbody>
    </table>
</div>

<!-- A template for time series item rows. -->
<script type="text/template" id="time-series-row">
<tr class="{{- error }}">
    <td class="time-series-date">{{- date }}</td>
    <td class="time-series-time">{{- time }}</td>
    <td class="time-series-speed">{{- speed }}</td>
    <td class="time-series-direction">{{- direction }} &deg;</td>
    <td>
    	<button type="button" class="btn btn-info btn-xs edit-time">
    	    <i class="glyphicon glyphicon-edit"></i>
    	</button>
    	<button type="button" class="btn btn-warning btn-xs delete-time">
    	    <i class="glyphicon glyphicon-trash"></i>
    	</button>
    </td>
</tr>
</script>