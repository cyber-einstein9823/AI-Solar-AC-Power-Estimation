/* Saved NumPy model, evaluated locally. The animated scene never alters inference. */
"use strict";
(() => {
  const fields = ["hour", "sw_radiation", "temp_2m", "cloud_cover"];
  const tabs = ["explorer", "evidence", "plots"];
  const model = window.SOLAR_MODEL;
  const $ = id => document.getElementById(id);
  const format = (value, digits = 1) => Number(value).toLocaleString("en-US", {minimumFractionDigits: digits, maximumFractionDigits: digits});
  const presets = {sunny:[13,950,35,10], cloudy:[13,400,28,85], sunset:[18,100,27,25], night:[0,0,24,15]};
  let current = {hour:13, sw_radiation:950, temp_2m:35, cloud_cover:10};
  let currentResult = null;

  function infer(values) {
    const feature = [values.sw_radiation, values.temp_2m, values.cloud_cover,
      Math.sin(2 * Math.PI * values.hour / 24), Math.cos(2 * Math.PI * values.hour / 24)];
    const scaled = feature.map((x,i) => (x-model.train_means[i])/model.train_stds[i]);
    const raw = scaled.reduce((total,x,i) => total+x*model.theta[i+1], model.theta[0]);
    return {raw, power:Math.max(0,raw), features:feature, scaled};
  }
  function scene(values) {
    const h=values.hour;
    const phase=h<6 || h>=19 ? "night" : h<=7 ? "dawn" : h>=17 ? "dusk" : "day";
    const weather=values.cloud_cover>=70 ? "cloudy" : values.cloud_cover>=25 ? "partly" : "clear";
    const sky=weather==="cloudy" ? "Overcast" : weather==="partly" ? "Partly cloudy" : "Clear";
    const time=phase==="night" ? "night" : phase==="dawn" ? "morning" : phase==="dusk" ? "evening" : h<12 ? "morning" : "afternoon";
    document.body.dataset.phase=phase;
    document.body.dataset.weather=weather;
    document.documentElement.style.setProperty("--sun-x", `${22+Math.max(0,Math.min(1,(h-6)/12))*62}%`);
    document.documentElement.style.setProperty("--sun-y", `${10+Math.abs(h-12)*3}%`);
    $("scene-label").textContent=`${sky} ${time}`;
    $("scene-time").textContent=`${String(h).padStart(2,"0")}:00 IST`;
    $("scene-symbol").textContent=phase==="night" ? "☾" : weather==="cloudy" ? "☁" : phase==="dawn" || phase==="dusk" ? "◒" : "☀";
    $("summary-sky").textContent=sky;
    $("summary-hour").textContent=`${String(h).padStart(2,"0")}:00`;
    $("summary-temp").textContent=`${format(values.temp_2m,1)} °C`;
  }
  function paintSlider(field) {
    const slider=$(`${field}-slider`);
    slider.style.setProperty("--range-fill",`${100*(Number(slider.value)-Number(slider.min))/(Number(slider.max)-Number(slider.min))}%`);
    if(field==="hour") slider.setAttribute("aria-valuetext",`${String(slider.value).padStart(2,"0")}:00 IST`);
  }
  function update() {
    const values={};
    let error="";
    for(const field of fields) {
      const input=$(`${field}-number`);
      const value=input.value.trim()==="" ? NaN : Number(input.value);
      const valid=Number.isFinite(value) && value>=Number(input.min) && value<=Number(input.max) && !input.validity.stepMismatch && (field!=="hour" || Number.isInteger(value));
      input.setAttribute("aria-invalid",String(!valid));
      if(!valid) error="Use whole numbers for hour (0–23), radiation (0–1,500) and cloud cover (0–100). Temperature must be −20–60 °C in 0.1-degree steps.";
      values[field]=value;
    }
    $("input-error").hidden=!error;
    $("input-error").textContent=error;
    $("download").disabled=Boolean(error);
    if(error) {
      $("power-value").textContent="—";
      $("power-mw").textContent="Check your inputs";
      currentResult=null;
      return;
    }
    current=values;
    for(const field of fields) { $(`${field}-slider`).value=values[field]; paintSlider(field); }
    currentResult=infer(values);
    $("power-value").textContent=format(currentResult.power);
    $("power-mw").textContent=`${format(currentResult.power/1000,3)} MW`;
    $("raw-prediction").textContent=`Current unclipped output: ${format(currentResult.raw,4)} kW. Displayed output: ${format(currentResult.power,1)} kW.`;
    const warnings=[];
    if((values.hour<6 || values.hour>=19) && values.sw_radiation>50) warnings.push("Nighttime with substantial sunlight may be inconsistent. Your inputs are retained; the model is not overridden.");
    if(values.sw_radiation===0 && currentResult.power>0) warnings.push("The linear model can predict positive power without sunlight. This is a model limitation, not observed generation.");
    if(model.feature_min && [values.sw_radiation,values.temp_2m,values.cloud_cover].some((v,i)=>v<model.feature_min[i] || v>model.feature_max[i])) warnings.push("One or more weather inputs are outside the training range; this is extrapolation.");
    $("scenario-warning").textContent=warnings.join(" ");
    $("scenario-warning").hidden=!warnings.length;
    scene(values);
    profile(values.hour);
  }
  function clearPreset() {
    document.querySelectorAll("[data-preset]").forEach(button=>{button.classList.remove("active");button.setAttribute("aria-pressed","false");});
  }
  function selectPreset(name) {
    fields.forEach((field,i)=>{$(`${field}-number`).value=presets[name][i];});
    clearPreset();
    const button=document.querySelector(`[data-preset="${name}"]`);
    button.classList.add("active"); button.setAttribute("aria-pressed","true");
    update();
  }
  function profile(hour) {
    const data=model.hourly_profile;
    const max=Math.max(...data)*1.16 || 1;
    const w=460,h=88;
    const points=data.map((v,i)=>[i/23*w,h-v/max*h]);
    const line=points.map(p=>p.map(v=>v.toFixed(2)).join(",")).join(" ");
    const [x,y]=points[hour];
    $("profile-chart").innerHTML=`<svg viewBox="-3 -5 466 99" role="img" aria-label="Historical mean AC power by hour in kilowatts. Selected hour ${hour}, mean ${format(data[hour])} kilowatts."><defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--accent)" stop-opacity=".2"/><stop offset="1" stop-color="var(--accent)" stop-opacity="0"/></linearGradient></defs><path d="M0,${h/2}H${w} M0,${h}H${w}" stroke="#bccbda19" fill="none" stroke-dasharray="3 5"/><polygon points="0,${h} ${line} ${w},${h}" fill="url(#area)"/><polyline points="${line}" stroke="var(--accent)" stroke-width="1.7" fill="none" stroke-linejoin="round"/><line x1="${x}" y1="0" x2="${x}" y2="${h}" stroke="var(--accent)" stroke-opacity=".3" stroke-dasharray="3 4"/><circle cx="${x}" cy="${y}" r="7" fill="var(--accent)" fill-opacity=".15"/><circle cx="${x}" cy="${y}" r="3" fill="var(--accent)"/></svg>`;
  }
  function evidence() {
    $("train-count").textContent=model.summary.train_rows;
    $("test-count").textContent=model.summary.test_rows;
    const names={normal:"Normal equation",batch_gd:"Batch GD",sgd:"Ordered SGD"};
    for(const metric of model.metrics) {
      const row=document.createElement("tr");
      [metric.feature_set==="A" ? "A · On-site sensors" : "B · Public weather",names[metric.solver],format(metric.rmse_all_hours,2),format(metric.rmse_daytime,2)].forEach(value=>{const td=document.createElement("td");td.textContent=value;row.appendChild(td);});
      $("metrics-rows").appendChild(row);
      if(metric.model==="B normal") { $("public-rmse").textContent=format(metric.rmse_daytime,1); $("plot-rmse-b").textContent=`${format(metric.rmse_daytime,2)} kW`; }
      if(metric.model==="A normal") $("plot-rmse-a").textContent=`${format(metric.rmse_daytime,2)} kW`;
    }
    const scoreA=model.metrics.find(row=>row.model==="A normal").rmse_daytime;
    const scoreB=model.metrics.find(row=>row.model==="B normal").rmse_daytime;
    $("plot-comparison").textContent=`Public-weather daytime RMSE is ${format(scoreB/scoreA,2)} times the on-site model's error in this held-out week. Both use the same daytime observations and normal-equation solver. Location and timing uncertainty means the public-weather comparison remains provisional.`;
    const series=model.test_series;
    const w=960,h=210,max=Math.max(...series.flatMap(row=>[row.actual,row.sensor,row.weather]))*1.08;
    const paths=[["actual","#dae8ec"],["sensor","#84d6bb"],["weather","#f0c989"]].map(([key,color])=>{
      const points=series.map((row,i)=>`${(i/(series.length-1)*w).toFixed(2)},${(h-row[key]/max*h).toFixed(2)}`).join(" ");
      return `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round"/>`;
    }).join("");
    $("test-chart").innerHTML=`<svg viewBox="0 -10 960 230" role="img" aria-label="Actual and predicted plant AC power in kilowatts over 11 to 17 June 2020; maximum plotted value ${format(max,0)} kilowatts"><path d="M0,${h/2}H960 M0,${h}H960" stroke="#ffffff20" stroke-dasharray="4 5"/>${paths}</svg>`;
  }
  function switchTab(name,focus=false) {
    for(const key of tabs) {
      const active=key===name;
      $(`${key}-tab`).setAttribute("aria-selected",String(active));
      $(`${key}-tab`).tabIndex=active ? 0 : -1;
      $(`${key}-panel`).hidden=!active;
    }
    if(focus) $(`${name}-tab`).focus();
  }
  function motion(paused) {
    document.body.classList.toggle("motion-paused",paused);
    $("motion-toggle").setAttribute("aria-pressed",String(paused));
    $("motion-toggle").textContent=paused ? "Resume motion" : "Pause motion";
    $("motion-toggle").title=paused ? "Resume ambient animation (system reduced-motion preference is respected)" : "Pause background animation";
  }
  for(let i=0;i<45;i++) {
    const star=document.createElement("span"); star.className="star";
    star.style.cssText=`left:${(i*43.71)%100}%;top:${(i*17.13)%70}%;width:${i%3===0?2:1}px;height:${i%3===0?2:1}px;animation-delay:-${i%7}s;animation-duration:${4+i%5}s`;
    $("stars").appendChild(star);
  }
  const reduced=window.matchMedia("(prefers-reduced-motion: reduce)");
  motion(reduced.matches);
  reduced.addEventListener("change",event=>motion(event.matches));
  $("motion-toggle").addEventListener("click",()=>motion(!document.body.classList.contains("motion-paused")));
  for(const name of tabs) {
    $(`${name}-tab`).addEventListener("click",()=>switchTab(name));
    $(`${name}-tab`).addEventListener("keydown",event=>{if(["ArrowLeft","ArrowRight","Home","End"].includes(event.key)){event.preventDefault();const index=tabs.indexOf(name);switchTab(event.key==="Home"?tabs[0]:event.key==="End"?tabs[tabs.length-1]:tabs[(index+(event.key==="ArrowRight"?1:-1)+tabs.length)%tabs.length],true);}});
  }
  $("evidence-link").addEventListener("click",event=>{event.preventDefault();switchTab("evidence",true);$("evidence-tab").scrollIntoView({block:"start"});});
  if(!model || !Array.isArray(model.theta) || model.theta.length!==6 || !model.theta.every(Number.isFinite) || !model.train_stds?.every(v=>Number.isFinite(v)&&v>0)) {
    $("input-error").hidden=false;
    $("input-error").textContent="Saved model data is missing or invalid. Run the project HTML server to rebuild model-data.js. No demonstration value will be substituted.";
    $("download").disabled=true;
    document.querySelector(".model-pill").textContent="MODEL UNAVAILABLE";
    return;
  }
  for(const field of fields) {
    $(`${field}-slider`).addEventListener("input",event=>{$(`${field}-number`).value=event.target.value;clearPreset();update();});
    $(`${field}-number`).addEventListener("input",()=>{clearPreset();update();});
  }
  document.querySelectorAll("[data-preset]").forEach(button=>button.addEventListener("click",()=>selectPreset(button.dataset.preset)));
  $("current-hour").addEventListener("click",()=>{
    const hour=new Intl.DateTimeFormat("en-GB",{timeZone:"Asia/Kolkata",hour:"2-digit",hourCycle:"h23"}).format(new Date());
    $("hour-number").value=Number(hour); clearPreset();update();
  });
  $("download").addEventListener("click",()=>{
    if(!currentResult) return;
    const csv="hour_IST,sw_radiation_W_m2,temperature_C,cloud_cover_percent,raw_prediction_kW,predicted_AC_kW\r\n"+fields.map(field=>current[field]).concat([currentResult.raw,currentResult.power]).join(",")+"\r\n";
    const url=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"}));
    const link=document.createElement("a");link.href=url;link.download="solar_scenario.csv";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  evidence();update();
})();
