/* xapi记录如下数据：
    `object.id`属性记录了用户当前所选定课程id，
    `verb.id`属性记录了用户对于当前课程推荐的每一门课是否跟随链接，如"recommend1"代表用户点击了相似推荐列表中的第一门课程，"recommend-3"代表用户点击了不相似推荐列表中的第三门课程
*/
var VERB_LIST = ["recommend1", "recommend2", "recommend3", "recommend4", "recommend5", 
                 "recommend-1", "recommend-2", "recommend-3", "recommend-4", "recommend-5"];
var OBJECT_LIST = [...Array(NUMBER_OF_COURSES).keys()];

function compose_xapi_request(mbox, actor, verb, object) {
    
    /* Json Scheme */
    var statement = {
        "actor": {
            "mbox": mbox,
            "name": actor,
            "objectType": "Agent"
        },
        "verb": {
            "id": verb,
            "display": { "en-US": verb }
        },
        "object": {
            "id": object,
            "definition": {
                "name": { "en-US": object },
                "description": { "en-US": object }
            },
            "objectType": "Activity"
        }
    };
    
    return statement;
}
