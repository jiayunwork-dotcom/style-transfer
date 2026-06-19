import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import time
import difflib

API_BASE = "http://localhost:8000/api"


def api_get(path, params=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API请求失败: {e}")
        return None


def api_post(path, data=None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API请求失败: {e}")
        return None


def api_delete(path):
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API请求失败: {e}")
        return None


def radar_chart(scores, title="评估评分"):
    categories = ["内容保持度", "风格强度", "流畅性"]
    values = [
        scores.get("content_preservation", 0),
        scores.get("style_intensity", 0),
        scores.get("fluency", 0),
    ]
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        name=title,
        line=dict(color="rgb(99,110,250)"),
        fillcolor="rgba(99,110,250,0.3)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=title,
        height=400,
    )
    return fig


def bar_chart_comparison(avg_a, avg_b, method_a_name, method_b_name):
    categories = ["内容保持度", "风格强度", "流畅性", "综合"]
    keys = ["content_preservation", "style_intensity", "fluency", "overall"]
    vals_a = [avg_a.get(k, 0) for k in keys]
    vals_b = [avg_b.get(k, 0) for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(name=method_a_name, x=categories, y=vals_a, marker_color="rgb(99,110,250)"))
    fig.add_trace(go.Bar(name=method_b_name, x=categories, y=vals_b, marker_color="rgb(239,85,59)"))
    fig.update_layout(barmode="group", title="A/B对比评分", yaxis=dict(range=[0, 1]), height=400)
    return fig


def multi_record_bar_chart(records):
    labels = [f"记录#{r['id']}" for r in records]
    categories = ["内容保持度", "风格强度", "流畅性"]
    keys = ["content_preservation", "style_intensity", "fluency"]
    colors = ["rgb(99,110,250)", "rgb(239,85,59)", "rgb(0,164,239)", "rgb(255,161,0)"]

    fig = go.Figure()
    for ci, (cat, key) in enumerate(zip(categories, keys)):
        vals = [r["scores"].get(key, 0) for r in records]
        fig.add_trace(go.Bar(name=cat, x=labels, y=vals, marker_color=colors[ci % len(colors)]))

    fig.update_layout(
        barmode="group",
        title="多维评分对比",
        yaxis=dict(range=[0, 1]),
        height=400,
    )
    return fig


def highlight_diff_html(text1, text2):
    sm = difflib.SequenceMatcher(None, text1, text2)
    parts1 = []
    parts2 = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts1.append(text1[i1:i2])
            parts2.append(text2[j1:j2])
        elif tag == "replace":
            parts1.append(f'<span style="background-color:#ffcccc;">{text1[i1:i2]}</span>')
            parts2.append(f'<span style="background-color:#ccffcc;">{text2[j1:j2]}</span>')
        elif tag == "delete":
            parts1.append(f'<span style="background-color:#ffcccc;">{text1[i1:i2]}</span>')
        elif tag == "insert":
            parts2.append(f'<span style="background-color:#ccffcc;">{text2[j1:j2]}</span>')
    return "".join(parts1), "".join(parts2)


def render_sidebar():
    st.sidebar.title("⚙️ 设置")
    styles = api_get("/styles")
    if not styles:
        st.sidebar.warning("无法获取风格列表")
        return {}, ""

    style_options = {}
    for s in styles:
        stype = s.get("style_type", "preset" if s.get("is_preset") else "custom")
        type_tag = {"preset": "", "custom": "[自定义]", "mixed": "[混合]"}.get(stype, "")
        label = f"{s['name']} ({s['key']}) {type_tag}".strip()
        style_options[label] = s["key"]
    selected_style_display = st.sidebar.selectbox("选择目标风格", list(style_options.keys()))
    selected_style_key = style_options[selected_style_display]

    method = st.sidebar.selectbox("选择迁移方式", ["rule", "prompt", "hybrid"],
                                   format_func=lambda x: {"rule": "规则迁移", "prompt": "提示词迁移", "hybrid": "混合迁移"}[x])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 评估权重配置")
    w_content = st.sidebar.slider("内容保持度权重", 0.0, 1.0, 0.4, 0.05)
    w_style = st.sidebar.slider("风格强度权重", 0.0, 1.0, 0.35, 0.05)
    w_fluency = st.sidebar.slider("流畅性权重", 0.0, 1.0, 0.25, 0.05)

    return {
        "style_key": selected_style_key,
        "method": method,
        "weights": {"content_preservation": w_content, "style_intensity": w_style, "fluency": w_fluency},
    }, selected_style_key


def page_single_migration(settings, style_key):
    st.header("📝 单文本风格迁移")

    if "compare_basket" not in st.session_state:
        st.session_state["compare_basket"] = set()
    if "compare_details" not in st.session_state:
        st.session_state["compare_details"] = {}
    if "show_comparison" not in st.session_state:
        st.session_state["show_comparison"] = False

    source_text = st.text_area("输入原文", height=150, placeholder="请输入需要风格迁移的文本...")

    if st.button("🚀 开始迁移", type="primary", disabled=not source_text.strip()):
        if not source_text.strip():
            st.warning("请输入文本")
            return

        with st.spinner("正在迁移..."):
            result = api_post("/migrate", {
                "text": source_text,
                "target_style": settings["style_key"],
                "method": settings["method"],
            })

        if result:
            st.success("迁移完成！")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("原文")
                st.text_area("", result["source_text"], height=150, disabled=True)
            with col2:
                st.subheader("改写结果")
                st.text_area("", result["result_text"], height=150, disabled=True)

            st.subheader("📊 评估评分")
            scores = result["scores"]
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("内容保持度", f"{scores['content_preservation']:.2%}")
            col_s2.metric("风格强度", f"{scores['style_intensity']:.2%}")
            col_s3.metric("流畅性", f"{scores['fluency']:.2%}")
            col_s4.metric("综合评分", f"{scores['overall']:.2%}")

            fig = radar_chart(scores, f"迁移方式: {settings['method']}")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    basket = st.session_state["compare_basket"]
    details = st.session_state["compare_details"]

    col_title, col_count, col_view, col_clear = st.columns([3, 2, 1, 1])
    with col_title:
        st.subheader("📜 最近迁移记录")
    with col_count:
        st.markdown(f"**对比篮**: {len(basket)}/4 条记录")
    with col_view:
        if len(basket) >= 2:
            if st.button("查看对比", type="primary", key="btn_view_compare"):
                st.session_state["show_comparison"] = True
    with col_clear:
        if len(basket) > 0:
            if st.button("清空", key="btn_clear_compare"):
                st.session_state["compare_basket"] = set()
                st.session_state["compare_details"] = {}
                st.session_state["show_comparison"] = False
                st.rerun()

    if st.session_state["show_comparison"] and len(basket) >= 2:
        st.markdown("---")
        st.subheader("📊 对比视图")

        compare_records = []
        for rid in sorted(basket):
            if str(rid) in details:
                compare_records.append(details[str(rid)])
            else:
                full = api_get(f"/result/{rid}")
                if full:
                    details[str(rid)] = full
                    compare_records.append(full)

        n = len(compare_records)
        cols = st.columns(n)
        for i, record in enumerate(compare_records):
            with cols[i]:
                method_label = {"rule": "规则迁移", "prompt": "提示词迁移", "hybrid": "混合迁移"}.get(record["method"], record["method"])
                st.markdown(f"#### 记录 #{record['id']}")
                st.markdown(f"**风格**: {record['target_style']}")
                st.markdown(f"**方式**: {method_label}")
                st.markdown("**原文**:")
                st.text_area("", record["source_text"], height=80, disabled=True, key=f"cmp_src_{record['id']}")
                st.markdown("**结果**:")
                st.text_area("", record["result_text"], height=80, disabled=True, key=f"cmp_res_{record['id']}")
                sc = record["scores"]
                st.caption(f"内容:{sc['content_preservation']:.2%} 风格:{sc['style_intensity']:.2%} 流畅:{sc['fluency']:.2%} 综合:{sc['overall']:.2%}")

        st.markdown("---")
        st.subheader("📈 评分柱状图对比")
        fig = multi_record_bar_chart(compare_records)
        st.plotly_chart(fig, use_container_width=True)

        same_source_groups = {}
        for record in compare_records:
            src = record["source_text"]
            same_source_groups.setdefault(src, []).append(record)

        diff_groups = {src: recs for src, recs in same_source_groups.items() if len(recs) >= 2}
        if diff_groups:
            st.markdown("---")
            st.subheader("🔍 差异高亮")
            st.markdown('<span style="background-color:#ffcccc;">红色底</span> = 记录A独有 &nbsp; <span style="background-color:#ccffcc;">绿色底</span> = 记录B独有', unsafe_allow_html=True)

            for src, recs in diff_groups.items():
                has_diff = any(
                    recs[0]["method"] != recs[j]["method"] or recs[0]["target_style"] != recs[j]["target_style"]
                    for j in range(1, len(recs))
                )
                if not has_diff:
                    continue

                st.markdown(f"**相同原文** (风格或方式不同):")
                with st.expander(f"原文: {src[:60]}...", expanded=True):
                    base = recs[0]
                    other_records = recs[1:]
                    for oi, other in enumerate(other_records):
                        method_label_base = {"rule": "规则迁移", "prompt": "提示词迁移", "hybrid": "混合迁移"}.get(base["method"], base["method"])
                        method_label_other = {"rule": "规则迁移", "prompt": "提示词迁移", "hybrid": "混合迁移"}.get(other["method"], other["method"])
                        label_a = f"记录#{base['id']} ({method_label_base}, {base['target_style']})"
                        label_b = f"记录#{other['id']} ({method_label_other}, {other['target_style']})"
                        html_a, html_b = highlight_diff_html(base["result_text"], other["result_text"])

                        col_da, col_db = st.columns(2)
                        with col_da:
                            st.markdown(f"**{label_a}**:")
                            st.markdown(f'<div style="border:1px solid #ddd; padding:8px; border-radius:4px; max-height:200px; overflow-y:auto; font-size:14px; line-height:1.6;">{html_a}</div>', unsafe_allow_html=True)
                        with col_db:
                            st.markdown(f"**{label_b}**:")
                            st.markdown(f'<div style="border:1px solid #ddd; padding:8px; border-radius:4px; max-height:200px; overflow-y:auto; font-size:14px; line-height:1.6;">{html_b}</div>', unsafe_allow_html=True)

        if st.button("关闭对比视图", key="btn_close_compare"):
            st.session_state["show_comparison"] = False
            st.rerun()

    history = api_get("/history", {"limit": 20})
    if history:
        for item in history:
            rid = item["id"]
            in_basket = rid in basket

            with st.expander(f"#{item['id']} | {item['target_style']} | {item['method']} | 综合: {item['scores']['overall']:.2%}"):
                st.markdown(f"**原文**: {item['source_text']}")
                st.markdown(f"**结果**: {item['result_text']}")
                st.json(item["scores"])

                col_cb, col_spacer = st.columns([1, 3])
                with col_cb:
                    if st.checkbox(
                        "加入对比",
                        value=in_basket,
                        key=f"cmp_cb_{rid}",
                        disabled=not in_basket and len(basket) >= 4,
                    ):
                        if rid not in basket:
                            if len(basket) >= 4:
                                st.warning("最多选择4条记录加入对比")
                            else:
                                basket.add(rid)
                                full = api_get(f"/result/{rid}")
                                if full:
                                    details[str(rid)] = full
                    else:
                        basket.discard(rid)
                        details.pop(str(rid), None)


def page_style_management():
    st.header("🎨 风格管理")

    tab_list, tab_create = st.tabs(["风格列表", "创建自定义风格"])

    with tab_list:
        styles = api_get("/styles")
        if styles:
            st.markdown(f"共 **{len(styles)}** 种风格（{sum(1 for s in styles if s.get('style_type') == 'preset' or s.get('is_preset'))} 种预置 / {sum(1 for s in styles if s.get('style_type') == 'custom')} 种自定义 / {sum(1 for s in styles if s.get('style_type') == 'mixed')} 种混合）")
            for s in styles:
                stype = s.get("style_type", "preset" if s.get("is_preset") else "custom")
                badge_map = {"preset": "🟢 预置", "custom": "🔵 自定义", "mixed": "🟣 混合"}
                badge = badge_map.get(stype, "🔵 自定义")
                title = f"{badge} **{s['name']}** ({s['key']})"
                if stype == "mixed":
                    title += f" — {s.get('ratio_a', 0):.0%} {s.get('source_style_a', '')} + {s.get('ratio_b', 0):.0%} {s.get('source_style_b', '')}"
                with st.expander(title):
                    st.markdown(f"**描述**: {s['description']}")
                    if s.get("features"):
                        st.markdown("**风格指纹特征**:")
                        feat_dict = s["features"] if isinstance(s["features"], dict) else json.loads(s["features"])
                        feat_df = pd.DataFrame([
                            {"特征": k, "数值": f"{v:.4f}" if isinstance(v, (int, float)) else v}
                            for k, v in feat_dict.items()
                        ])
                        st.dataframe(feat_df, use_container_width=True)
        else:
            st.warning("暂无风格数据")

    with tab_create:
        st.markdown("### 🌟 创建新风格")
        st.info("💡 请提供风格的关键信息，并输入至少3段该风格的示例文本，系统将自动提取量化指纹。")

        col_k, col_n = st.columns(2)
        with col_k:
            style_key = st.text_input("风格标识 (Key，英文，如 tech_report)", placeholder="请输入风格标识，如 tech_report")
        with col_n:
            style_name = st.text_input("风格名称 (中文)", placeholder="请输入风格名称，如 技术报告")

        style_desc = st.text_area("风格描述", placeholder="请简要描述该风格的特点和适用场景...")

        st.markdown("#### 📝 示例文本（至少3段）")
        example_1 = st.text_area("示例文本 1", height=80, placeholder="请输入第一段示例文本...")
        example_2 = st.text_area("示例文本 2", height=80, placeholder="请输入第二段示例文本...")
        example_3 = st.text_area("示例文本 3", height=80, placeholder="请输入第三段示例文本...")
        example_4 = st.text_area("示例文本 4 (可选)", height=80, placeholder="可输入更多示例以提高指纹精度...")
        example_5 = st.text_area("示例文本 5 (可选)", height=80, placeholder="可输入更多示例以提高指纹精度...")

        if st.button("✅ 创建自定义风格", type="primary", disabled=not (style_key and style_name and style_desc and example_1 and example_2 and example_3)):
            if not style_key.strip():
                st.error("请输入风格标识")
                return
            if not style_name.strip():
                st.error("请输入风格名称")
                return

            examples = [example_1.strip(), example_2.strip(), example_3.strip()]
            for ex in [example_4, example_5]:
                if ex.strip():
                    examples.append(ex.strip())

            if len(examples) < 3:
                st.error("请输入至少3段示例文本")
                return

            with st.spinner("正在分析示例文本并生成风格指纹..."):
                result = api_post("/styles/create", {
                    "key": style_key.strip(),
                    "name": style_name.strip(),
                    "description": style_desc.strip(),
                    "example_texts": examples,
                })

            if result:
                st.success(f"🎉 风格 '{result['name']}' 创建成功！")
                st.markdown("**提取的风格指纹**:")
                feat_dict = result.get("features", {})
                if feat_dict:
                    feat_df = pd.DataFrame([
                        {"特征": k, "数值": f"{v:.4f}" if isinstance(v, (int, float)) else v}
                        for k, v in feat_dict.items()
                    ])
                    st.dataframe(feat_df, use_container_width=True)
                st.balloons()


def page_batch_processing(settings, style_key):
    st.header("📦 批量处理")

    texts_input = st.text_area("输入文本列表（每行一条）", height=200,
                                placeholder="请输入需要批量迁移的文本，每行一条...")

    if st.button("📤 提交批量任务", type="primary", disabled=not texts_input.strip()):
        texts = [t.strip() for t in texts_input.split("\n") if t.strip()]
        if len(texts) > 100:
            st.error("单次最多100条文本")
            return
        if not texts:
            st.warning("请输入至少一条文本")
            return

        result = api_post("/batch/submit", {
            "texts": texts,
            "target_style": settings["style_key"],
            "method": settings["method"],
        })
        if result:
            st.success(f"批量任务已提交！任务ID: {result['task_id']}, 总数: {result['total']}")
            st.session_state["batch_task_id"] = result["task_id"]

    st.markdown("---")

    task_id_input = st.number_input("输入任务ID查询进度", min_value=1, step=1,
                                     value=st.session_state.get("batch_task_id", 1))

    col_q, col_c = st.columns(2)
    with col_q:
        if st.button("🔍 查询进度"):
            progress = api_get(f"/batch/progress/{task_id_input}")
            if progress:
                st.json(progress)
                pct = progress.get("progress", 0)
                st.progress(pct)
                if progress["status"] == "running":
                    st.info(f"已完成 {progress['completed']}/{progress['total']}")
                    if progress.get("estimated_remaining_seconds"):
                        st.info(f"预计剩余时间: {progress['estimated_remaining_seconds']:.1f}秒")
                elif progress["status"] == "completed":
                    st.success("任务已完成！")
                elif progress["status"] == "cancelled":
                    st.warning("任务已取消")
                elif progress["status"] == "failed":
                    st.error("任务失败")

    with col_c:
        if st.button("❌ 取消任务"):
            result = api_post(f"/batch/cancel/{task_id_input}", {})
            if result:
                st.warning(result.get("message", "已取消"))


def page_ab_comparison(settings, style_key):
    st.header("⚖️ A/B对比")

    tab_create, tab_results = st.tabs(["创建A/B任务", "查看结果"])

    with tab_create:
        ab_name = st.text_input("任务名称", value="A/B对比测试")
        col_ma, col_mb = st.columns(2)
        with col_ma:
            method_a = st.selectbox("方式A", ["rule", "prompt", "hybrid"],
                                     format_func=lambda x: {"rule": "规则迁移", "prompt": "提示词迁移", "hybrid": "混合迁移"}[x],
                                     key="ab_method_a")
        with col_mb:
            method_b = st.selectbox("方式B", ["prompt", "rule", "hybrid"],
                                     format_func=lambda x: {"rule": "规则迁移", "prompt": "提示词迁移", "hybrid": "混合迁移"}[x],
                                     key="ab_method_b")

        ab_texts = st.text_area("输入对比文本（每行一条）", height=150,
                                 placeholder="请输入用于A/B对比的文本...")

        if st.button("🔄 创建A/B任务", type="primary", disabled=not ab_texts.strip()):
            texts = [t.strip() for t in ab_texts.split("\n") if t.strip()]
            if not texts:
                st.warning("请输入至少一条文本")
                return
            if method_a == method_b:
                st.error("方式A和方式B不能相同")
                return

            with st.spinner("正在执行A/B对比任务..."):
                result = api_post("/ab/create", {
                    "name": ab_name,
                    "method_a": method_a,
                    "method_b": method_b,
                    "target_style": settings["style_key"],
                    "texts": texts,
                })
            if result:
                st.success(f"A/B任务已创建！任务ID: {result['task_id']}")
                st.session_state["ab_task_id"] = result["task_id"]

    with tab_results:
        ab_tasks = api_get("/ab/tasks")
        if ab_tasks:
            task_options = {f"#{t['id']} {t['name']} ({t['method_a']} vs {t['method_b']})": t["id"] for t in ab_tasks}
            selected_task = st.selectbox("选择A/B任务", list(task_options.keys()))
            task_id = task_options[selected_task]

            if st.button("📊 查看对比结果"):
                results = api_get(f"/ab/results/{task_id}")
                if results:
                    st.subheader("评分对比")
                    avg_a = results["avg_scores_a"]
                    avg_b = results["avg_scores_b"]
                    method_a_name = results["method_a"]
                    method_b_name = results["method_b"]

                    fig = bar_chart_comparison(avg_a, avg_b, method_a_name, method_b_name)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(f"**{method_a_name}** 平均分: 综合 {avg_a.get('overall', 0):.2%}")
                    st.markdown(f"**{method_b_name}** 平均分: 综合 {avg_b.get('overall', 0):.2%}")

                    diff = results["score_diff"]
                    st.markdown(f"**差异**: 内容 {diff['content_preservation']:+.2%} | 风格 {diff['style_intensity']:+.2%} | 流畅 {diff['fluency']:+.2%}")

                    st.subheader("逐条对比")
                    for i, pair in enumerate(results["pairs"]):
                        with st.expander(f"文本 #{i+1}"):
                            st.markdown(f"**原文**: {pair['source_text']}")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown(f"**方式A ({method_a_name})**:")
                                st.write(pair["result_a"])
                                if pair.get("scores_a"):
                                    st.caption(f"内容:{pair['scores_a'].get('content_preservation',0):.2%} "
                                               f"风格:{pair['scores_a'].get('style_intensity',0):.2%} "
                                               f"流畅:{pair['scores_a'].get('fluency',0):.2%}")
                            with col_b:
                                st.markdown(f"**方式B ({method_b_name})**:")
                                st.write(pair["result_b"])
                                if pair.get("scores_b"):
                                    st.caption(f"内容:{pair['scores_b'].get('content_preservation',0):.2%} "
                                               f"风格:{pair['scores_b'].get('style_intensity',0):.2%} "
                                               f"流畅:{pair['scores_b'].get('fluency',0):.2%}")

                    st.subheader("🗳️ 人工偏好投票")
                    annotator = st.text_input("标注员名称", key="ab_annotator")
                    if annotator:
                        for i, pair in enumerate(results["pairs"]):
                            st.markdown(f"**文本 #{i+1}**: {pair['source_text'][:50]}...")
                            col_pref_a, col_pref_b = st.columns(2)
                            with col_pref_a:
                                if st.button(f"👍 偏好方式A-{i+1}", key=f"pref_a_{i}"):
                                    pref_result = api_post("/ab/preference", {
                                        "task_id": task_id,
                                        "annotator": annotator,
                                        "source_text": pair["source_text"],
                                        "preferred_method": method_a_name,
                                    })
                                    if pref_result:
                                        st.success("已记录偏好")
                            with col_pref_b:
                                if st.button(f"👍 偏好方式B-{i+1}", key=f"pref_b_{i}"):
                                    pref_result = api_post("/ab/preference", {
                                        "task_id": task_id,
                                        "annotator": annotator,
                                        "source_text": pair["source_text"],
                                        "preferred_method": method_b_name,
                                    })
                                    if pref_result:
                                        st.success("已记录偏好")

                        prefs = api_get(f"/ab/preferences/{task_id}")
                        if prefs and prefs.get("total", 0) > 0:
                            st.markdown(f"**偏好统计**: {prefs['method_a']} {prefs['a_rate']:.1%} vs "
                                        f"{prefs['method_b']} {prefs['b_rate']:.1%} (共{prefs['total']}票)")
        else:
            st.info("暂无A/B对比任务")


RADAR_FEATURE_LABELS = {
    "avg_sentence_length": "平均句长",
    "long_sentence_ratio": "长句比例",
    "passive_voice_ratio": "被动语态比例",
    "colloquial_ratio": "口语化比例",
    "terminology_density": "术语密度",
    "avg_formality": "正式度",
}


def _style_radar_traces(features, name, color, fill_color):
    categories = list(RADAR_FEATURE_LABELS.values())
    keys = list(RADAR_FEATURE_LABELS.keys())
    values = []
    for k in keys:
        v = features.get(k, 0)
        values.append(v)
    values.append(values[0])
    categories.append(categories[0])
    return go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        name=name,
        line=dict(color=color),
        fillcolor=fill_color,
    )


def _normalize_features_for_radar(features):
    max_vals = {
        "avg_sentence_length": 50,
        "long_sentence_ratio": 1.0,
        "passive_voice_ratio": 1.0,
        "colloquial_ratio": 1.0,
        "terminology_density": 1.0,
        "avg_formality": 5.0,
    }
    normalized = {}
    for k, v in features.items():
        if k in max_vals:
            normalized[k] = min(v / max_vals[k], 1.0)
        else:
            normalized[k] = v
    return normalized


def page_style_mixing():
    st.header("🧪 风格混合实验")

    tab_mix, tab_manage = st.tabs(["混合实验", "混合风格管理"])

    with tab_mix:
        styles = api_get("/styles")
        if not styles:
            st.warning("无法获取风格列表，请确认后端服务已启动")
            return

        non_mixed_styles = [s for s in styles if s.get("style_type") != "mixed"]
        if len(non_mixed_styles) < 2:
            st.warning("至少需要2种非混合风格才能进行混合实验")
            return

        style_options = {f"{s['name']} ({s['key']})": s["key"] for s in non_mixed_styles}
        style_features_map = {s["key"]: s.get("features", {}) for s in non_mixed_styles}

        col_a, col_b = st.columns(2)
        with col_a:
            selected_a_display = st.selectbox("选择源风格 A", list(style_options.keys()), key="mix_style_a")
            style_a_key = style_options[selected_a_display]
        with col_b:
            remaining_options = {k: v for k, v in style_options.items() if v != style_a_key}
            if not remaining_options:
                remaining_options = style_options
            selected_b_display = st.selectbox("选择源风格 B", list(remaining_options.keys()), key="mix_style_b")
            style_b_key = style_options[selected_b_display]

        ratio = st.slider("混合比例 (风格A占比)", 0, 100, 70, step=1) / 100.0
        st.markdown(f"**混合配比**: {ratio:.0%} 风格A + {1 - ratio:.0%} 风格B")

        features_a = style_features_map.get(style_a_key, {})
        features_b = style_features_map.get(style_b_key, {})

        preview_data = api_post("/mix/preview", {
            "source_style_a": style_a_key,
            "source_style_b": style_b_key,
            "ratio_a": ratio,
        })

        if preview_data:
            mixed_features = preview_data.get("features", {})
        else:
            mixed_features = {}
            for k in RADAR_FEATURE_LABELS.keys():
                val_a = features_a.get(k, 0)
                val_b = features_b.get(k, 0)
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    mixed_features[k] = ratio * val_a + (1 - ratio) * val_b

        st.subheader("📊 风格特征雷达图对比")
        norm_a = _normalize_features_for_radar(features_a)
        norm_b = _normalize_features_for_radar(features_b)
        norm_mixed = _normalize_features_for_radar(mixed_features)

        fig = go.Figure()
        fig.add_trace(_style_radar_traces(norm_a, "风格A", "rgb(99,110,250)", "rgba(99,110,250,0.15)"))
        fig.add_trace(_style_radar_traces(norm_b, "风格B", "rgb(239,85,59)", "rgba(239,85,59,0.15)"))
        fig.add_trace(_style_radar_traces(norm_mixed, "混合风格", "rgb(0,164,239)", "rgba(0,164,239,0.3)"))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="风格特征对比（归一化）",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 混合风格特征参数详情"):
            if mixed_features:
                feat_df = pd.DataFrame([
                    {
                        "特征": RADAR_FEATURE_LABELS.get(k, k),
                        "风格A": f"{features_a.get(k, 0):.4f}" if isinstance(features_a.get(k, 0), (int, float)) else str(features_a.get(k, 0)),
                        "风格B": f"{features_b.get(k, 0):.4f}" if isinstance(features_b.get(k, 0), (int, float)) else str(features_b.get(k, 0)),
                        "混合结果": f"{v:.4f}" if isinstance(v, (int, float)) else str(v),
                    }
                    for k, v in mixed_features.items()
                ])
                st.dataframe(feat_df, use_container_width=True)

        st.markdown("---")
        st.subheader("🚀 保存并执行混合迁移")

        col_name, col_key = st.columns(2)
        with col_name:
            mix_name = st.text_input("混合风格名称", value=f"{ratio:.0%}A+{1 - ratio:.0%}B混合", key="mix_name")
        with col_key:
            mix_key = st.text_input("混合风格Key", value=f"mix_{style_a_key}_{style_b_key}_{int(ratio*100)}", key="mix_key")

        mix_desc = st.text_input("混合风格描述（可选）", value="", key="mix_desc")

        text_input = st.text_area("输入待迁移文本", height=120, placeholder="请输入需要风格迁移的文本...", key="mix_text_input")

        col_save, col_migrate = st.columns(2)
        with col_save:
            if st.button("💾 保存混合风格", type="secondary", disabled=not (mix_key and mix_name)):
                result = api_post("/mix/create", {
                    "key": mix_key.strip(),
                    "name": mix_name.strip(),
                    "description": mix_desc.strip(),
                    "source_style_a": style_a_key,
                    "source_style_b": style_b_key,
                    "ratio_a": ratio,
                })
                if result:
                    st.success(f"混合风格 '{result['name']}' 保存成功！")
                else:
                    st.error("保存失败，Key可能已存在")

        with col_migrate:
            if st.button("🎯 执行混合迁移", type="primary", disabled=not (text_input.strip() and mix_key)):
                if not text_input.strip():
                    st.warning("请输入待迁移文本")
                else:
                    existing = api_get("/mix/list")
                    found = False
                    if existing:
                        found = any(m["key"] == mix_key.strip() for m in existing)

                    if not found:
                        save_result = api_post("/mix/create", {
                            "key": mix_key.strip(),
                            "name": mix_name.strip(),
                            "description": mix_desc.strip(),
                            "source_style_a": style_a_key,
                            "source_style_b": style_b_key,
                            "ratio_a": ratio,
                        })
                        if not save_result:
                            st.error("保存混合风格失败，无法执行迁移")
                            return

                    with st.spinner("正在执行混合风格迁移..."):
                        migrate_result = api_post("/mix/migrate", {
                            "text": text_input.strip(),
                            "mixed_style_key": mix_key.strip(),
                        })

                    if migrate_result:
                        st.success("混合迁移完成！")

                        col_orig, col_res = st.columns(2)
                        with col_orig:
                            st.subheader("原文")
                            st.text_area("", migrate_result["source_text"], height=120, disabled=True, key="mix_result_orig")
                        with col_res:
                            st.subheader("迁移结果")
                            st.text_area("", migrate_result["result_text"], height=120, disabled=True, key="mix_result_res")

                        st.subheader("📊 三维评分")
                        scores = migrate_result["scores"]
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        col_s1.metric("内容保持度", f"{scores['content_preservation']:.2%}")
                        col_s2.metric("风格强度", f"{scores['style_intensity']:.2%}")
                        col_s3.metric("流畅性", f"{scores['fluency']:.2%}")
                        col_s4.metric("综合评分", f"{scores['overall']:.2%}")

                        fig = radar_chart(scores, "混合迁移评分")
                        st.plotly_chart(fig, use_container_width=True)

                        info = migrate_result
                        st.info(f"混合风格: {info.get('mixed_style_name', '')} | "
                                f"源A: {info.get('source_a', '')} ({info.get('ratio_a', 0):.0%}) | "
                                f"源B: {info.get('source_b', '')} ({info.get('ratio_b', 0):.0%})")

    with tab_manage:
        st.subheader("📂 混合风格列表")
        mixed_list = api_get("/mix/list")
        if mixed_list:
            st.markdown(f"共 **{len(mixed_list)}** 种混合风格")
            for m in mixed_list:
                with st.expander(f"🟣 **{m['name']}** ({m['key']}) — {m['ratio_a']:.0%} {m['source_style_a']} + {m['ratio_b']:.0%} {m['source_style_b']}"):
                    st.markdown(f"**描述**: {m['description']}")
                    if m.get("features"):
                        feat_dict = m["features"]
                        feat_df = pd.DataFrame([
                            {"特征": RADAR_FEATURE_LABELS.get(k, k), "数值": f"{v:.4f}" if isinstance(v, (int, float)) else v}
                            for k, v in feat_dict.items()
                        ])
                        st.dataframe(feat_df, use_container_width=True)

                    col_del, col_spacer = st.columns([1, 3])
                    with col_del:
                        if st.button(f"🗑️ 删除", key=f"del_mix_{m['key']}"):
                            del_result = api_delete(f"/mix/delete/{m['key']}")
                            if del_result:
                                st.success(f"混合风格 '{m['key']}' 已删除")
                                st.rerun()
        else:
            st.info("暂无混合风格，请在'混合实验'标签页中创建")


def main():
    st.set_page_config(page_title="文本风格迁移系统", page_icon="✨", layout="wide")
    st.title("✨ 文本风格迁移与改写质量评估系统")

    settings, style_key = render_sidebar()

    tab_single, tab_style, tab_batch, tab_ab, tab_mix = st.tabs([
        "📝 单文本迁移", "🎨 风格管理", "📦 批量处理", "⚖️ A/B对比", "🧪 风格混合实验"
    ])

    with tab_single:
        page_single_migration(settings, style_key)
    with tab_style:
        page_style_management()
    with tab_batch:
        page_batch_processing(settings, style_key)
    with tab_ab:
        page_ab_comparison(settings, style_key)
    with tab_mix:
        page_style_mixing()


if __name__ == "__main__":
    main()
