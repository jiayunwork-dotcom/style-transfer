import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import time

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


def render_sidebar():
    st.sidebar.title("⚙️ 设置")
    styles = api_get("/styles")
    if not styles:
        st.sidebar.warning("无法获取风格列表")
        return {}, ""

    style_options = {f"{s['name']} ({s['key']})": s["key"] for s in styles}
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
    st.subheader("📜 最近迁移记录")
    history = api_get("/history", {"limit": 20})
    if history:
        for item in history:
            with st.expander(f"#{item['id']} | {item['target_style']} | {item['method']} | 综合: {item['scores']['overall']:.2%}"):
                st.markdown(f"**原文**: {item['source_text']}")
                st.markdown(f"**结果**: {item['result_text']}")
                st.json(item["scores"])


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
                                if st.button(f"👍 偏好方式X-1", key=f"pref_a_{i}"):
                                    pref_result = api_post("/ab/preference", {
                                        "task_id": task_id,
                                        "annotator": annotator,
                                        "source_text": pair["source_text"],
                                        "preferred_method": method_a_name,
                                    })
                                    if pref_result:
                                        st.success("已记录偏好")
                            with col_pref_b:
                                if st.button(f"👍 偏好方式X-2", key=f"pref_b_{i}"):
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


def main():
    st.set_page_config(page_title="文本风格迁移系统", page_icon="✨", layout="wide")
    st.title("✨ 文本风格迁移与改写质量评估系统")

    settings, style_key = render_sidebar()

    tab_single, tab_batch, tab_ab = st.tabs(["📝 单文本迁移", "📦 批量处理", "⚖️ A/B对比"])

    with tab_single:
        page_single_migration(settings, style_key)
    with tab_batch:
        page_batch_processing(settings, style_key)
    with tab_ab:
        page_ab_comparison(settings, style_key)


if __name__ == "__main__":
    main()
